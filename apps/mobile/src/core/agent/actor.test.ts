import { AIMessage, type BaseMessage, HumanMessage } from '@langchain/core/messages';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { CloudSocket } from '@/core/net/cloud-socket';
import { QueueChatModel } from '@/core/net/queue-chat-model';
import type { DoneEvent, LlmRequest, ServerEvent } from '@/core/protocol/messages';
import type { Db } from '@/core/storage/db';
import { readStoryMeta, type StoryMeta } from '@/core/story/story-meta';
import type { Entity } from '@/core/world/model';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { actorSystemMessage, SILENCE } from './actor';
import { ChatStore, GM_CHAT } from './chat-store';
import { createGmAgent } from './gm-agent';
import { queueModels } from './models';
import { isUnfinished, retryTurn, runTurn } from './run-turn';
import { conversation, lastToolText } from './testing/demo-scenario';
import { ScriptedChatModel, type Step } from './testing/scripted-model';
import { TranscriptStore } from './transcript';

const TAVERN = 'loc:rusty-gear-tavern';
const LAMPLIGHT = 'loc:lamplight-row';
const MAGS = 'npc:mags-holloway';
const PIP = 'npc:pip-quillon';
const CHARACTER = { id: 'c1', name: 'Aria', sex: null, age: null, race: 'human', description: 'A cartographer.' };

let env: TestEnv;
let db: Db;
let world: World;
let meta: StoryMeta;
let chats: ChatStore;
let transcript: TranscriptStore;
let master: ScriptedChatModel;
/** Lines of the actor pack by entity id, in order; every call is recorded with who was asked. */
let lines: Map<string, Step[]>;
let actorCalls: { entity: Entity; messages: BaseMessage[]; tools: string[] }[];

beforeEach(async () => {
  env = await makeTestEnv();
  await env.library.importer.import(DEMO_STORY);
  await env.library.repository.addCharacter({ ...CHARACTER, createdAt: '2026-10-15T00:00:00.000Z' });
  const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: 'c1', title: 't' });
  db = await env.library.saves.open(save.id);
  world = await World.open(db);
  const started = await world.startSession({ storyId: DEMO_STORY_ID, character: CHARACTER, startLocation: TAVERN });
  if (!started.ok) throw new Error(started.error);
  meta = await readStoryMeta(db);
  chats = new ChatStore(db);
  transcript = new TranscriptStore(db);
  master = new ScriptedChatModel([]);
  lines = new Map();
  actorCalls = [];
});

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

/** The entity with this id will answer with these lines in the actor pack, in order. */
function says(id: string, ...steps: (Step | string)[]) {
  lines.set(id, [...(lines.get(id) ?? []), ...steps.map((s) => (typeof s === 'string' ? { text: s } : s))]);
}

const actorModel = (entity: Entity) => {
  const model = new ScriptedChatModel([]);
  return {
    invoke: async (messages: BaseMessage[]) => {
      const next = lines.get(entity.id)?.shift();
      if (!next) throw new Error(`${entity.id} was not expected to react`);
      actorCalls.push({ entity, messages, tools: model.boundTools });
      model.push(next);
      return model.invoke(messages);
    },
  } as unknown as ScriptedChatModel;
};

const agentFor = () => createGmAgent({ model: master, world, meta, actor: { model: actorModel, chats } });

function play(input: string, ...steps: Step[]) {
  master.push(...steps);
  return runTurn({ agent: agentFor(), world, chats, transcript, input });
}

const react = (target: string, situation: string) => ({ toolCalls: [{ name: 'react_entity', args: { target, situation } }] });
const call = (name: string, args: Record<string, unknown>) => ({ name, args });
const promptOf = (i: number) => actorCalls[i].messages[0].content as { text: string; cache_control?: unknown }[];

describe('react_entity and the actor pack (ADR 45)', () => {
  it('Mags speaks for herself from her card, the scene and her own chat; the master answers after', async () => {
    const line = '*She taps the plan.* "Below the bank, love. That\'s all you get."';
    says(MAGS, line);
    const result = await play(
      'I ask Mags about the vault.',
      react('Mags Holloway', 'The new recruit asks you about the vault.'),
      (messages) => {
        expect(lastToolText(messages)).toBe('Mags Holloway [npc:mags-holloway] reacts now.');
        // her line, shown as hers; nothing added after it (ADR 46: the cached prefix holds)
        expect(conversation(messages).at(-1)!.text).toBe(`Mags Holloway [npc:mags-holloway]: ${line}`);
        expect(master.boundTools).not.toContain('actor_note');
        return { text: 'The tavern noise swallows the rest.' };
      },
    );
    expect(result.reply).toBe('The tavern noise swallows the rest.');
    expect(result.events.map((e) => [e.kind, e.payload])).toEqual([
      ['entity.reacted', { entity: MAGS, situation: 'The new recruit asks you about the vault.', reaction: line }],
    ]);
    expect((await transcript.lines()).map((l) => [l.speaker, l.text.slice(0, 18), l.extra])).toEqual([
      ['player', 'I ask Mags about t', {}],
      [MAGS, '*She taps the plan', { name: 'Mags Holloway' }],
      ['gm', 'The tavern noise s', {}],
    ]);

    // one small call: her card and the scene, the situation; no tools, no master history
    expect(actorCalls).toHaveLength(1);
    const [system, ask] = actorCalls[0].messages;
    expect(actorCalls[0].messages).toHaveLength(2);
    expect(ask.text).toBe('The new recruit asks you about the vault.');
    const [stable, scene] = system.content as { text: string; cache_control?: unknown }[];
    expect(stable.cache_control).toEqual({ type: 'ephemeral' });
    expect(stable.text).toMatch(/^You are Mags Holloway, in an interactive story\./);
    expect(stable.text).toContain('a brass hook where her left hand used to be');
    expect(stable.text).toContain('- Silas Ashcombe: revenge on Silas Ashcombe, who seized her airship');
    expect(stable.text).toContain('answer exactly: [silence]');
    expect(scene.text).toContain('You are in The Rusty Gear Tavern');
    expect(scene.text).toContain('Facing you: the player character, Aria (human) — A cartographer.');

    // her chat keeps the exchange; the master's history keeps the line, marked as hers
    expect((await chats.load(MAGS)).map((m) => [m.getType(), m.text])).toEqual([
      ['human', 'The new recruit asks you about the vault.'],
      ['ai', line],
    ]);
    const saved = (await chats.load(GM_CHAT)).find((m) => m.additional_kwargs?.qq_actor);
    expect(saved?.text).toBe(line);
    expect(saved?.additional_kwargs?.qq_actor).toEqual({ id: MAGS, name: 'Mags Holloway' });
  });

  it('remembers the conversation and her notes in the next actor pack', async () => {
    says(MAGS, '"Below the bank."', '"I told you: below the bank."');
    await play('Where is the vault?', react(MAGS, 'The recruit asks where the vault is.'), { text: '.' });
    expectOk(await world.addActorNote(MAGS, 'I do not trust the recruit yet.'));
    await play('Where again?', react(MAGS, 'The recruit asks again.'), { text: '.' });
    expect(actorCalls[1].messages.slice(1).map((m) => m.text)).toEqual([
      'The recruit asks where the vault is.',
      '"Below the bank."',
      'The recruit asks again.',
    ]);
    // her chat ends with a cache breakpoint (ADR 49)
    expect((actorCalls[1].messages[2].content as { cache_control?: unknown }[])[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(promptOf(1)[1].text).toContain('# What you remember\n- I do not trust the recruit yet.');
    expect((await world.scene()).notes).toEqual([]); // her notes are not the master's
  });

  it('a magic mirror in the inventory speaks too', async () => {
    says('item:magic-mirror', '*The glass ripples.* "Ask me about the keeper of the vault."');
    const result = await play(
      'I pull out the mirror I found and look into it.',
      {
        toolCalls: [
          call('set_entity', { name: 'magic mirror', type: 'item', description: 'A hand mirror that remembers every face it has shown.' }),
        ],
      },
      { toolCalls: [call('set_relation', { a: 'player', relation: 'has_item', b: 'magic mirror' })] },
      react('magic mirror', 'The stranger who carries you looks into you.'),
      { text: 'Your reflection blinks late.' },
    );
    expect(result.events.map((e) => e.kind)).toEqual(['entity.added', 'item.taken', 'entity.reacted']);
    expect(promptOf(0)[0].text).toContain('remembers every face');
    expect(promptOf(0)[1].text).toContain('The player character carries you.');
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', 'npc:item:magic-mirror', 'gm']);
  });

  it('a stone stays silent: no bubble, the master is told it does not react', async () => {
    says('item:crate', SILENCE);
    const result = await play('I ask the crate where the gold is.', react('crate', 'Someone asks you where the gold is.'), (messages) => {
      expect(conversation(messages).at(-1)!.text).toBe('crate [item:crate] does not react.');
      return { text: 'The crate keeps its counsel.' };
    });
    expect(result.events.map((e) => e.payload)).toEqual([
      { entity: 'item:crate', situation: 'Someone asks you where the gold is.', reaction: SILENCE },
    ]);
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', 'gm']);
    expect((await chats.load('npc:item:crate')).map((m) => m.text)).toEqual(['Someone asks you where the gold is.', SILENCE]);
  });

  it('asks an innkeeper the master just added to the scene', async () => {
    says('npc:innkeeper', '"Ale\'s a penny. Questions cost two."');
    const result = await play(
      'I wave over whoever runs this place.',
      {
        toolCalls: [
          call('set_entity', { name: 'innkeeper', type: 'npc', description: 'A bald man polishing mugs that never get clean.' }),
          call('react_entity', { target: 'innkeeper', situation: 'A stranger waves you over.' }),
        ],
      },
      { text: 'He waits.' },
    );
    expect(result.events.map((e) => e.kind)).toEqual(['entity.added', 'entity.reacted']);
    expect(promptOf(0)[0].text).toContain('A bald man polishing mugs');
    expect(promptOf(0)[0].text).not.toContain('# Your ties');
    expect(promptOf(0)[1].text).toContain('Also here: Mags Holloway.');
  });

  it('two react in one reply: each speaks in turn, the master answers once after both', async () => {
    says(MAGS, '"No."');
    says('npc:innkeeper', '*He wipes the table, pointedly.*');
    await play(
      'I ask Mags and the innkeeper for a room.',
      {
        toolCalls: [
          call('set_entity', { name: 'innkeeper', type: 'npc', description: 'A bald man polishing mugs.' }),
          call('react_entity', { target: MAGS, situation: 'The stranger asks you for a room.' }),
          call('react_entity', { target: 'innkeeper', situation: 'The stranger asks for a room.' }),
        ],
      },
      (messages) => {
        expect(conversation(messages).slice(-2).map((m) => m.text)).toEqual([
          'Mags Holloway [npc:mags-holloway]: "No."',
          'innkeeper [npc:innkeeper]: *He wipes the table, pointedly.*',
        ]);
        return { text: 'No room tonight.' };
      },
    );
    expect(actorCalls.map((c) => c.entity.id)).toEqual([MAGS, 'npc:innkeeper']);
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', MAGS, 'npc:innkeeper', 'gm']);
  });

  it('snatching an item: the owner reacts first, then take succeeds (ADR 32, 34)', async () => {
    await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT);
    says(PIP, '*Pip lets go with a scowl.* "Keep it. It opens no doors without me."');
    const result = await play(
      "I snatch Pip's calling card.",
      react('Pip Quillon', 'The stranger snatches the calling card from your fingers.'),
      { toolCalls: [call('set_relation', { a: 'player', relation: 'has_item', b: 'calling card' })] },
      (messages) => ({ text: lastToolText(messages).startsWith('The player now has calling card') ? 'Yours now.' : '?' }),
    );
    expect(result.reply).toBe('Yours now.');
    expect(result.events.map((e) => [e.kind, e.payload])).toEqual([
      ['entity.reacted', expect.objectContaining({ entity: PIP })],
      ['item.taken', { item: 'doc:calling-card', from: PIP }],
    ]);
  });

  it('snatching an item: take in the same reply as the reaction is refused', async () => {
    await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT);
    says(PIP, '*Pip twists away and is gone into the crowd.*');
    const result = await play(
      "I snatch Pip's calling card.",
      {
        toolCalls: [
          call('set_relation', { a: 'player', relation: 'has_item', b: 'calling card' }),
          call('react_entity', { target: PIP, situation: 'The stranger grabs at your calling card.' }),
        ],
      },
      { text: 'Your fingers close on air.' },
    );
    const results = result.messages.filter((m) => m.getType() === 'tool').map((m) => m.text);
    expect(results[0]).toMatch(/^Error: Pip Quillon has calling card\. Play out the attempt first: call react_entity/);
    expect(result.events.map((e) => e.kind)).toEqual(['entity.reacted']);
    expect((await world.state.inventory(PIP)).map((e) => e.id)).toEqual(['doc:calling-card']);
  });

  it('refuses something that is not here; a failing actor model becomes an error for the master', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const result = await play(
      'I shout for Silas Ashcombe.',
      react('Silas Ashcombe', 'Someone shouts your name.'),
      react('Mags Holloway', 'The recruit shouts for Silas Ashcombe.'),
      (messages) => {
        expect(conversation(messages).at(-1)!.text).toMatch(/^Error: Mags Holloway \[npc:mags-holloway\] could not react right now \(npc:mags-holloway was not expected/);
        return { text: 'Nobody answers.' };
      },
    );
    const results = result.messages.filter((m) => m.getType() === 'tool').map((m) => m.text);
    expect(results[0]).toBe('Error: there is no "Silas Ashcombe" here to react.');
    expect(result.events).toEqual([]);
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', 'gm']);
    expect(await chats.load(MAGS)).toEqual([]);
  });

  it('caps a long entity chat in the actor pack (the background notes fold it later)', async () => {
    const filler = 'w'.repeat(4_000);
    await chats.append(
      MAGS,
      Array.from({ length: 10 }, (_, i) => [new HumanMessage(`Q${i} ${filler}`), new AIMessage(`A${i} ${filler}`)]).flat(),
    );
    says(MAGS, '"What were we saying?"');
    await play('Hi, Mags.', react(MAGS, 'The recruit says hi.'), { text: '.' });
    const sent = actorCalls[0].messages.slice(1);
    expect(sent.length).toBeLessThan(21);
    expect(sent.at(-2)!.text).toMatch(/^A9 /);
    expect(sent.at(-1)!.text).toBe('The recruit says hi.');
  });
});

describe('a draft the Guard added can react (ADR 48)', () => {
  it('the bartender, just noticed, speaks in this turn and stays', async () => {
    says('npc:bartender', '"What\'ll it be?"');
    const guardModel = new ScriptedChatModel([
      {
        text: JSON.stringify({
          verdict: 'pass',
          reason: 'r',
          reply: '',
          expected: [{ name: 'bartender', type: 'npc', description: 'A broad man behind the bar.', likelihood: 'certain' }],
        }),
      },
    ]);
    master.push(react('bartender', 'A stranger waves you over for an ale.'), { text: 'He reaches for a mug.' });
    const agent = createGmAgent({ model: master, world, meta, guardModel, actor: { model: actorModel, chats } });
    const result = await runTurn({ agent, world, chats, transcript, input: 'I wave the bartender over.' });
    expect(result.events.map((e) => e.kind)).toEqual(['entity.added', 'entity.reacted', 'draft.confirmed']);
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', 'npc:bartender', 'gm']);
    expect((await world.state.entity('npc:bartender'))?.props.draft).toBeUndefined();
  });
});

describe('a turn cut in the middle (retryTurn)', () => {
  const retry = () => retryTurn({ agent: agentFor(), world, chats, transcript });

  it('cut after the line: the master goes on, the line is not asked again', async () => {
    says(MAGS, '"Shove off."');
    await expect(
      play('Hi, Mags.', react(MAGS, 'The recruit says hi.'), () => {
        throw new Error('network');
      }),
    ).rejects.toThrow('network');
    const saved = await chats.load(GM_CHAT);
    expect(saved.at(-1)!.additional_kwargs?.qq_actor).toEqual({ id: MAGS, name: 'Mags Holloway' });
    expect(isUnfinished(saved)).toBe(true);

    master.push((messages) => ({
      text: conversation(messages).at(-1)!.text === 'Mags Holloway [npc:mags-holloway]: "Shove off."' ? 'She turns away.' : '?',
    }));
    const result = await retry();
    expect(result.reply).toBe('She turns away.');
    expect(actorCalls).toHaveLength(1);
    expect(result.events.map((e) => e.kind)).toEqual(['entity.reacted']);
    expect((await transcript.lines()).map((l) => [l.speaker, l.text])).toEqual([
      ['player', 'Hi, Mags.'],
      [MAGS, '"Shove off."'],
      ['gm', 'She turns away.'],
    ]);
  });

  it('cut after the reaction was journaled but before the line was saved: it is not asked twice', async () => {
    says(MAGS, '"Shove off."');
    await expect(
      play('Hi, Mags.', react(MAGS, 'The recruit says hi.'), () => {
        throw new Error('network');
      }),
    ).rejects.toThrow('network');
    // simulate the lost line: drop it from the master's saved history
    const saved = await chats.load(GM_CHAT);
    await chats.writeFrom(GM_CHAT, 0, []);
    await chats.append(GM_CHAT, saved.filter((m) => !m.additional_kwargs?.qq_actor));

    master.push({ text: 'She turns away.' });
    const result = await retry();
    expect(actorCalls).toHaveLength(1); // the line came back from her chat
    expect(result.events.map((e) => e.kind)).toEqual(['entity.reacted']);
    expect((await chats.load(MAGS))).toHaveLength(2);
  });
});

describe('actor stream', () => {
  it('the line streams on npc:<id> with her name, purpose npc, without tools', async () => {
    const requests: LlmRequest[] = [];
    const listeners = new Set<(event: ServerEvent) => void>();
    const emit = (event: ServerEvent) => listeners.forEach((listener) => listener(event));
    const socket = {
      onEvent(listener: (event: ServerEvent) => void) {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
      async request(request: LlmRequest): Promise<DoneEvent> {
        requests.push(request);
        const routed = { v: 1 as const, requestId: request.requestId, sessionId: request.sessionId, channel: request.channel };
        const text = request.purpose === 'npc' ? '"Who\'s asking?"' : 'Mags eyes you.';
        emit({ ...routed, type: 'delta', seq: 0, text });
        const calls =
          request.purpose === 'gm' && requests.filter((r) => r.purpose === 'gm').length === 1
            ? [
                {
                  id: 'call_1',
                  type: 'function' as const,
                  function: { name: 'react_entity', arguments: JSON.stringify({ target: MAGS, situation: 'The recruit greets you.' }) },
                },
              ]
            : undefined;
        return {
          ...routed,
          type: 'done',
          message: { role: 'assistant', content: calls ? null : text, ...(calls ? { tool_calls: calls } : {}) },
          finishReason: calls ? 'tool_calls' : 'stop',
          usage: { input: 10, output: 5, cacheRead: 0, costUsd: 0 },
        };
      },
    } as unknown as CloudSocket;
    const seen: ServerEvent[] = [];
    socket.onEvent((event) => seen.push(event));

    const models = queueModels(new QueueChatModel({ socket, sessionId: 'save-1', channel: 'gm', purpose: 'gm' }));
    const agent = createGmAgent({ ...models, world, meta, actor: { model: models.actorModel, chats } });
    const result = await runTurn({ agent, world, chats, transcript, input: 'Hello, Mags.' });

    const actor = requests.filter((r) => r.purpose === 'npc');
    expect(actor).toHaveLength(1);
    expect(actor[0]).toMatchObject({ channel: MAGS, purpose: 'npc', meta: { speaker: 'Mags Holloway' }, sessionId: 'save-1' });
    // the client names a model class, never a model (ADR 50)
    expect(requests.every((r) => r.model === 'basic')).toBe(true);
    expect(actor[0].tools).toBeUndefined();
    expect(requests.filter((r) => r.purpose === 'gm')).toHaveLength(2);
    expect(seen.some((e) => e.type === 'delta' && e.channel === MAGS)).toBe(true);
    expect(result.reply).toBe('Mags eyes you.');
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', MAGS, 'gm']);
  });
});

function expectOk<T>(outcome: { ok: true; value: T } | { ok: false; error: string }): T {
  if (!outcome.ok) throw new Error(outcome.error);
  return outcome.value;
}

describe('a world that follows the player (ADR 42)', () => {
  it('what Mags tells about, once made real, can be taken (here the master makes it)', async () => {
    says(MAGS, '"Tools? Under the bench by the stairs. A canvas bag, mind the chisels."');
    await play(
      'I ask Mags where I can find tools for the lock.',
      react('Mags Holloway', 'The stranger asks where to find tools for the lock.'),
      {
        toolCalls: [
          call('set_entity', {
            name: 'bag of tools',
            type: 'item',
            description: 'A canvas bag under the bench by the stairs, heavy with chisels and picks.',
          }),
        ],
      },
      { text: 'She nods towards the stairs.' },
    );
    const result = await play(
      'I take the bag of tools from under the bench.',
      { toolCalls: [call('set_relation', { a: 'player', relation: 'has_item', b: 'bag of tools' })] },
      (messages) => ({ text: lastToolText(messages).startsWith('The player now has bag of tools') ? 'Heavy.' : '?' }),
    );
    expect(result.reply).toBe('Heavy.');
    expect(result.events.map((e) => e.kind)).toEqual(['item.taken']);
    expect((await world.scene()).inventory.map((e) => e.id)).toEqual(['item:bag-of-tools']);
  });

  it('lets an NPC leave for another location, where she is found later', async () => {
    const result = await play(
      'I tell Mags to wait for me on Lamplight Row.',
      { toolCalls: [call('set_relation', { a: 'Mags Holloway', relation: 'located_in', b: 'Lamplight Row' })] },
      (messages) => {
        expect(lastToolText(messages)).toBe(
          'Mags Holloway [npc:mags-holloway] is now in Lamplight Row [loc:lamplight-row].',
        );
        return { text: 'She shoulders her coat and goes up the stairs.' };
      },
    );
    expect(result.events.map((e) => e.kind)).toEqual(['npc.moved']);
    expect((await world.scene()).here.map((e) => e.id)).not.toContain(MAGS);

    await play('I follow her.', { toolCalls: [call('set_relation', { a: 'player', relation: 'located_in', b: 'Lamplight Row' })] }, { text: 'Gas lamps.' });
    expect((await world.scene()).here.map((e) => e.id)).toContain(MAGS);
    says(MAGS, '"Took your time."');
    await play('Hello again.', react(MAGS, 'The stranger catches up with you.'), { text: '.' });
    expect((promptOf(0))[1].text).toContain('You are in Lamplight Row');
  });

  it('after the player refuses the heist, Mags acts on the new tie, not on the planned one', async () => {
    const before = await actorSystemMessage(world, (await world.state.entity(MAGS))!);
    expect((before.content as { text: string }[])[0].text).toContain('- Pip Quillon: leader of the crew');

    const result = await play(
      "I won't join your heist, and Pip is coming with me.",
      {
        toolCalls: [
          call('set_relation', {
            a: 'Mags Holloway',
            relation: 'interacts_with',
            b: 'Aria',
            description: 'the stranger refused to join the heist and took Pip; she no longer trusts them',
          }),
          call('cancel_relation', { a: 'Mags Holloway', b: 'Pip Quillon' }),
          call('set_entity', {
            id: 'Mags Holloway',
            description: 'A former airship captain with a brass hook. Her crew fell apart; she plans alone now.',
          }),
        ],
      },
      { text: 'Mags says nothing. Her hook taps the table.' },
    );
    expect(result.events.map((e) => e.kind)).toEqual(['relation.set', 'relation.cancelled', 'entity.edited']);

    says(MAGS, '"Walk away, then."');
    await play('I nod to Mags.', react(MAGS, 'The stranger who turned you down nods to you.'), { text: '.' });
    const stable = (promptOf(0))[0].text;
    expect(stable).toContain('- Aria: the stranger refused to join the heist and took Pip; she no longer trusts them');
    expect(stable).not.toContain('leader of the crew');
    expect(stable).not.toContain('recruited the player');
    expect(stable).toContain('Her crew fell apart; she plans alone now.');
  });
});
