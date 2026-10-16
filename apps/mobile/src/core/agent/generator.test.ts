import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, type BaseMessage } from '@langchain/core/messages';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { Db } from '@/core/storage/db';
import { readStoryMeta, type StoryMeta } from '@/core/story/story-meta';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { generate, GeneratorAnswerError, generatorMessages, type WorldChanges } from './generator';
import { ScriptedChatModel } from './testing/scripted-model';
import { TurnInProgressError, exclusive } from './turn-lock';

const TAVERN = 'loc:rusty-gear-tavern';
const LAMPLIGHT = 'loc:lamplight-row';
const MAGS = 'npc:mags-holloway';
const PIP = 'npc:pip-quillon';
const PLAYER = 'player';
const CHARACTER = { id: 'c1', name: 'Aria', description: 'A cartographer.' };
const TURN = {
  turn: 1,
  input: "I won't join your heist, Mags.",
  reply: 'Mags sets down her mug. "Then we are strangers again."',
};

let env: TestEnv;
let db: Db;
let world: World;
let meta: StoryMeta;

beforeEach(async () => {
  env = await makeTestEnv();
  await env.library.importer.import(DEMO_STORY);
  const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 't' });
  db = await env.library.saves.open(save.id);
  world = await World.open(db);
  const started = await world.startSession({ storyId: DEMO_STORY_ID, character: CHARACTER, startLocation: TAVERN });
  if (!started.ok) throw new Error(started.error);
  meta = await readStoryMeta(db);
  world.beginTurn();
});

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

const changes = (c: Partial<WorldChanges>): string =>
  JSON.stringify({ relations: [], cancelled: [], moves: [], edits: [], additions: [], ...c });

const scripted = (answer: string) => new ScriptedChatModel([{ text: answer }]);

describe('generator', () => {
  it('reads the turn and the people in it, with the story first for the cache', async () => {
    expectOk(
      await world.recordReaction({ entity: MAGS, situation: 'The stranger refuses.', reaction: '"Suit yourself."' }),
    );
    const [system, context] = await generatorMessages({ world, meta }, TURN);

    const [stable] = system.content as { text: string; cache_control?: unknown }[];
    expect(stable.cache_control).toEqual({ type: 'ephemeral' });
    expect(stable.text).toMatch(/^You keep the world of an interactive story true to what happens in it\./);
    expect(stable.text).toContain(`Title: ${meta.title}`);
    expect(context.text).toContain('# World state\nPlayer: Aria [player].');
    expect(context.text).toContain('# People involved\n- Mags Holloway [npc:mags-holloway] (npc), in The Rusty Gear');
    expect(context.text).toContain('  - Pip Quillon [npc:pip-quillon]: leader of the crew');
    expect(context.text).toContain(`Player: ${TURN.input}`);
    expect(context.text).toContain(`Game Master: ${TURN.reply}`);
    expect(context.text).toContain('What people said and did:\n- Mags Holloway: "Suit yourself."');
    expect(context.text).toContain('Already recorded this turn:\n- nothing');
  });

  it('applies the changes through the world, journaled for the generator', async () => {
    const model = scripted(
      changes({
        cancelled: [{ a: MAGS, b: PIP }],
        relations: [{ a: MAGS, b: PLAYER, description: 'refused to join the heist; not to be trusted' }],
        moves: [{ who: PIP, to: TAVERN }],
        edits: [{ target: MAGS, description: 'A former airship captain. Her crew is gone; she plans alone.' }],
        additions: [{ name: 'Overturned Stool', type: 'other', description: 'Kicked over in the argument.', where: '' }],
      }),
    );
    const result = await generate({ model, world, meta }, TURN);

    expect(result.rejected).toEqual([]);
    expect(result.applied.map((e) => [e.kind, e.actor, e.turn])).toEqual([
      ['relation.cancelled', 'generator', 1],
      ['relation.set', 'generator', 1],
      ['npc.moved', 'generator', 1],
      ['entity.edited', 'generator', 1],
      ['entity.added', 'generator', 1],
    ]);
    const ties = (await world.state.ties(MAGS)).map((t) => [t.other.id, t.description]);
    expect(ties).toContainEqual([PLAYER, 'refused to join the heist; not to be trusted']);
    expect(ties.map(([id]) => id)).not.toContain(PIP);
    expect((await world.state.locationOf(PIP))?.id).toBe(TAVERN);
    expect((await world.state.entity(MAGS))?.description).toContain('she plans alone');
    expect((await world.scene()).here.map((e) => e.id)).toContain('ent:overturned-stool');
    // the master's own commands are journaled for the master again
    expectOk(await world.addNote('After the generator.'));
    expect((await world.log.recent(1))[0].actor).toBe('gm');
  });

  it('"the door in the alley": what an honest NPC said is there appears where she said (ADR 45)', async () => {
    expectOk(
      await world.recordReaction({
        entity: MAGS,
        situation: 'The stranger asks how to get into the bank unseen.',
        reaction: '"Lamplight Row. Service door behind the barrels. Knock twice."',
      }),
    );
    const [system, context] = await generatorMessages({ world, meta }, TURN);
    expect(system.text).toContain('# Look for gaps');
    expect(system.text).toContain('a legendary golden sword, no');
    expect(context.text).toContain('What people said and did:\n- Mags Holloway: "Lamplight Row. Service door');

    const model = scripted(
      changes({
        additions: [
          { name: 'service door', type: 'other', description: 'A narrow door behind the barrels; two knocks open it.', where: LAMPLIGHT },
          { name: 'golden sword', type: 'item', description: 'Legendary.', where: 'loc:nowhere' },
        ],
      }),
    );
    const result = await generate({ model, world, meta }, TURN);
    expect(result.applied.map((e) => e.kind)).toEqual(['entity.added', 'item.moved']);
    expect(result.rejected).toEqual([{ change: 'add golden sword in loc:nowhere', error: 'there is no loc:nowhere' }]);
    expect((await world.state.locationOf('ent:service-door'))?.id).toBe(LAMPLIGHT);
    expect(await world.state.entity('item:golden-sword')).toBeUndefined();
    expect((await world.scene()).here.map((e) => e.id)).not.toContain('ent:service-door');
  });

  it('drops the changes the world refuses and keeps the rest', async () => {
    const model = scripted(
      changes({
        relations: [{ a: 'npc:nobody', b: MAGS, description: 'x' }],
        moves: [{ who: MAGS, to: TAVERN }], // already there
        edits: [
          { target: PLAYER, description: 'A hero.' }, // the master keeps the player
          { target: PIP, description: 'Nervous now.' },
        ],
      }),
    );
    const result = await generate({ model, world, meta }, TURN);

    expect(result.applied.map((e) => e.kind)).toEqual(['entity.edited']);
    expect(result.rejected.map((r) => r.change)).toEqual([
      `relation npc:nobody — ${MAGS}`,
      `move ${MAGS} → ${TAVERN}`,
      `edit ${PLAYER}`,
    ]);
    expect(result.rejected[0].error).toMatch(/no "npc:nobody"/);
  });

  it('changes nothing on an empty answer, and fails on one it cannot read', async () => {
    expect(await generate({ model: scripted(changes({})), world, meta }, TURN)).toEqual({ applied: [], rejected: [] });
    await expect(generate({ model: scripted('Sure! Mags is sad now.'), world, meta }, TURN)).rejects.toBeInstanceOf(
      GeneratorAnswerError,
    );
    expect(await world.log.count()).toBe(1); // only session.started
  });

  it('holds the turn lock while it works, and gives up at its time limit', async () => {
    const hanging = {
      invoke: (_messages: BaseMessage[], options: { signal: AbortSignal }) =>
        new Promise<AIMessage>((_resolve, reject) =>
          options.signal.addEventListener('abort', () => reject(options.signal.reason)),
        ),
    } as unknown as BaseChatModel;
    const running = generate({ model: hanging, world, meta, timeoutMs: 50 }, TURN);
    await expect(exclusive(world, async () => 'a turn')).rejects.toBeInstanceOf(TurnInProgressError);
    await expect(running).rejects.toThrow();
    expect(await exclusive(world, async () => 'a turn')).toBe('a turn');
  });

  it('reads moves to anywhere: a person sent out of sight', async () => {
    const model = scripted(changes({ moves: [{ who: 'Mags Holloway', to: 'Lamplight Row' }] }));
    const result = await generate({ model, world, meta }, TURN);
    expect(result.applied[0].payload).toEqual({ npc: MAGS, from: TAVERN, to: LAMPLIGHT });
  });
});

function expectOk<T>(outcome: { ok: true; value: T } | { ok: false; error: string }): T {
  if (!outcome.ok) throw new Error(outcome.error);
  return outcome.value;
}
