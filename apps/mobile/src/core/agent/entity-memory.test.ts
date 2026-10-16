import { AIMessage, type BaseMessage, HumanMessage } from '@langchain/core/messages';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { Db } from '@/core/storage/db';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { ChatStore } from './chat-store';
import { ENTITY_CHAT_WINDOW_TOKENS, noteEntities } from './entity-memory';
import { estimateTokens } from './messages';
import { ScriptedChatModel } from './testing/scripted-model';

const TAVERN = 'loc:rusty-gear-tavern';
const MAGS = 'npc:mags-holloway';
const PIP = 'npc:pip-quillon';

let env: TestEnv;
let db: Db;
let world: World;
let chats: ChatStore;
let turn: number;

beforeEach(async () => {
  env = await makeTestEnv();
  await env.library.importer.import(DEMO_STORY);
  const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 't' });
  db = await env.library.saves.open(save.id);
  world = await World.open(db);
  const started = await world.startSession({
    storyId: DEMO_STORY_ID,
    character: { id: 'c1', name: 'Aria', description: 'A cartographer.' },
    startLocation: TAVERN,
  });
  if (!started.ok) throw new Error(started.error);
  chats = new ChatStore(db);
  turn = world.beginTurn();
});

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

/** `pairs` exchanges of about `size` characters each in the entity's chat, and its reaction this turn. */
async function talked(id: string, pairs: number, size = 1_000) {
  const filler = 'w'.repeat(size);
  await chats.append(
    id,
    Array.from({ length: pairs }, (_, i) => [new HumanMessage(`Q${i} ${filler}`), new AIMessage(`A${i} ${filler}`)]).flat(),
  );
  const reacted = await world.recordReaction({ entity: id, situation: 's', reaction: 'r' });
  if (!reacted.ok) throw new Error(reacted.error);
}

function notesModel(answer: unknown) {
  const model = new ScriptedChatModel([{ text: JSON.stringify(answer) }]);
  return model;
}

describe('noteEntities (note_entity)', () => {
  it('makes no call while every chat fits its window', async () => {
    await talked(MAGS, 3);
    const model = new ScriptedChatModel([]);
    expect(await noteEntities({ model, world, chats }, turn)).toEqual({ noted: [], folded: [], rejected: [] });
    expect(model.calls).toEqual([]);
  });

  it('folds the oldest exchanges of all overflowing chats in one call, keeping the latest', async () => {
    await talked(MAGS, 12);
    await talked(PIP, 12);
    expect(estimateTokens(await chats.load(MAGS))).toBeGreaterThan(ENTITY_CHAT_WINDOW_TOKENS);
    expectOk(await world.addActorNote(MAGS, 'I want the Clockheart.'));
    const model = notesModel({
      notes: [
        { entity: MAGS, text: 'I asked the recruit about locks; they dodged.' },
        { entity: 'Pip Quillon', text: 'The stranger owes me a drink.' },
        { entity: 'npc:silas-ashcombe', text: 'Not one of them.' },
      ],
    });
    const result = await noteEntities({ model, world, chats }, turn);

    // one request for both, with what each remembers and what each is about to forget
    expect(model.calls).toHaveLength(1);
    const [system, batch] = model.calls[0] as BaseMessage[];
    expect(system.text).toMatch(/^You keep the memories of characters/);
    expect(batch.text).toContain('## Mags Holloway [npc:mags-holloway]');
    expect(batch.text).toContain('Notes: I want the Clockheart.');
    expect(batch.text).toContain('## Pip Quillon [npc:pip-quillon]');
    expect(batch.text).toContain('- Happened: Q0 ');
    expect(batch.text).toContain('- Mags Holloway: A0 ');

    expect(result.folded).toEqual([MAGS, PIP]);
    expect(result.rejected).toEqual([{ entity: 'npc:silas-ashcombe', error: 'not one of the characters given' }]);
    expect((await world.log.actorNotes(MAGS, 10)).map((n) => n.text)).toEqual([
      'I want the Clockheart.',
      'I asked the recruit about locks; they dodged.',
    ]);
    expect((await world.log.actorNotes(PIP, 10)).map((n) => n.text)).toEqual(['The stranger owes me a drink.']);

    const kept = await chats.load(MAGS);
    expect(estimateTokens(kept)).toBeLessThanOrEqual(2_000);
    expect(kept[0].getType()).toBe('human');
    expect(kept.at(-1)!.text).toMatch(/^A11 /);
    expect(batch.text).not.toContain('A11 '); // the latest stay in the chat, not in the batch
  });

  it('folds nothing when the call fails: the chat waits for the next turn', async () => {
    await talked(MAGS, 12);
    const before = await chats.load(MAGS);
    await expect(noteEntities({ model: notesModel('not json'), world, chats }, turn)).rejects.toThrow();
    expect(await chats.load(MAGS)).toHaveLength(before.length);
    expect(await world.log.actorNotes(MAGS, 10)).toEqual([]);
  });
});

function expectOk<T>(outcome: { ok: true; value: T } | { ok: false; error: string }): T {
  if (!outcome.ok) throw new Error(outcome.error);
  return outcome.value;
}
