import type { BaseMessage } from '@langchain/core/messages';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { CloudSocket } from '@/core/net/cloud-socket';
import { QueueChatModel } from '@/core/net/queue-chat-model';
import type { DoneEvent, LlmRequest, ServerEvent } from '@/core/protocol/messages';
import type { Db } from '@/core/storage/db';
import { readStoryMeta, type StoryMeta } from '@/core/story/story-meta';
import type { Entity } from '@/core/world/model';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { ChatStore, GM_CHAT } from './chat-store';
import { createGmAgent } from './gm-agent';
import { queueModels } from './models';
import { npcChatId, SILENCE } from './npc';
import { runTurn } from './run-turn';
import { lastToolText } from './testing/demo-scenario';
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
/** NPC answers by NPC id; every call is recorded with who was asked. */
let npcScripts: Map<string, ScriptedChatModel>;
let npcCalls: { npc: Entity; messages: BaseMessage[] }[];

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
  npcScripts = new Map();
  npcCalls = [];
});

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

/** The NPC with this id will answer these lines, in order. */
function npcSays(id: string, ...lines: string[]) {
  const model = npcScripts.get(id) ?? new ScriptedChatModel([]);
  model.push(...lines.map((text) => ({ text })));
  npcScripts.set(id, model);
}

const npcModel = (npc: Entity) => {
  const scripted = npcScripts.get(npc.id);
  if (!scripted) throw new Error(`${npc.id} was not expected to react`);
  return {
    invoke: async (messages: BaseMessage[]) => {
      npcCalls.push({ npc, messages });
      return scripted.invoke(messages);
    },
  } as unknown as ScriptedChatModel;
};

function play(input: string, ...steps: Step[]) {
  master.push(...steps);
  const agent = createGmAgent({ model: master, world, meta, npc: { model: npcModel, chats, transcript } });
  return runTurn({ agent, world, chats, transcript, input });
}

const react = (target: string, situation: string) => ({ toolCalls: [{ name: 'npc_react', args: { target, situation } }] });

describe('npc_react', () => {
  it('asks a story NPC: persona and ties in her prompt, her own chat, the transcript, the journal', async () => {
    npcSays(MAGS, '*She taps the plan.* "Below the bank, love. Behind a lock that thinks."');
    const result = await play(
      'I ask Mags about the vault.',
      react('Mags Holloway', 'The new recruit asks you about the vault.'),
      (messages) => {
        const told = lastToolText(messages);
        expect(told).toMatch(/^Mags Holloway \[npc:mags-holloway\] reacted; the player already saw it, so do not repeat/);
        expect(told).toContain('Behind a lock that thinks.');
        return { text: 'The tavern noise swallows the rest.' };
      },
    );
    expect(result.reply).toBe('The tavern noise swallows the rest.');
    expect(result.events.map((e) => [e.kind, e.payload])).toEqual([
      [
        'npc.reacted',
        {
          npc: MAGS,
          situation: 'The new recruit asks you about the vault.',
          reaction: '*She taps the plan.* "Below the bank, love. Behind a lock that thinks."',
        },
      ],
    ]);

    const [system, situation] = npcCalls[0].messages;
    const [stable, scene] = system.content as { text: string; cache_control?: unknown }[];
    expect(stable.cache_control).toEqual({ type: 'ephemeral' });
    expect(stable.text).toMatch(/^You are Mags Holloway, a character in an interactive story\./);
    expect(stable.text).toContain('a brass hook where her left hand used to be');
    expect(stable.text).toContain('- Silas Ashcombe: revenge on Silas Ashcombe, who seized her airship');
    expect(stable.text).toContain('answer exactly: [silence]');
    expect(scene.text).toContain('You are in The Rusty Gear Tavern');
    expect(scene.text).toContain('Facing you: the player character, Aria (human) — A cartographer.');
    expect(situation.text).toBe('The new recruit asks you about the vault.');

    expect((await chats.load(npcChatId(npcCalls[0].npc))).map((m) => m.getType())).toEqual(['human', 'ai']);
    expect((await transcript.lines()).map((l) => [l.speaker, l.text.slice(0, 18), l.extra])).toEqual([
      ['player', 'I ask Mags about t', {}],
      [MAGS, '*She taps the plan', { name: 'Mags Holloway' }],
      ['gm', 'The tavern noise s', {}],
    ]);
  });

  it('remembers the conversation: the next question comes with the earlier exchange', async () => {
    npcSays(MAGS, '"Below the bank."', '"I told you: below the bank."');
    await play('Where is the vault?', react(MAGS, 'The recruit asks where the vault is.'), { text: '.' });
    await play('Where again?', react(MAGS, 'The recruit asks again.'), { text: '.' });
    expect(npcCalls[1].messages.slice(1).map((m) => m.text)).toEqual([
      'The recruit asks where the vault is.',
      '"Below the bank."',
      'The recruit asks again.',
    ]);
    // the master's chat does not get the NPC's chat, only its tool result
    expect((await chats.load(GM_CHAT)).some((m) => m.text === '"Below the bank."')).toBe(false);
  });

  it('asks an innkeeper the master just added to the scene', async () => {
    npcSays('npc:innkeeper', '"Ale\'s a penny. Questions cost two."');
    const result = await play(
      'I wave over whoever runs this place.',
      {
        toolCalls: [
          {
            name: 'add_entity',
            args: { name: 'innkeeper', type: 'npc', description: 'A bald man polishing mugs that never get clean.' },
          },
          { name: 'npc_react', args: { target: 'innkeeper', situation: 'A stranger waves you over.' } },
        ],
      },
      { text: 'He waits.' },
    );
    expect(result.events.map((e) => e.kind)).toEqual(['entity.added', 'npc.reacted']);
    const [stable, scene] = npcCalls[0].messages[0].content as { text: string }[];
    expect(stable.text).toContain('A bald man polishing mugs');
    expect(stable.text).not.toContain('# Your ties');
    expect(scene.text).toContain('Also here: Mags Holloway.');
    expect(scene.text).toContain('You carry: nothing of note.');
  });

  it('asks a stone: [silence] reaches neither the player nor the master as words', async () => {
    npcSays('item:crate', SILENCE);
    const result = await play('I ask the crate where the gold is.', react('crate', 'Someone asks you where the gold is.'), (messages) => {
      expect(lastToolText(messages)).toBe('crate [item:crate] does not react.');
      return { text: 'The crate keeps its counsel.' };
    });
    expect(result.events.map((e) => e.payload)).toEqual([
      { npc: 'item:crate', situation: 'Someone asks you where the gold is.', reaction: SILENCE },
    ]);
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', 'gm']);
    expect((await chats.load('npc:item:crate')).map((m) => m.text)).toEqual(['Someone asks you where the gold is.', SILENCE]);
  });

  it('snatching an item: the owner reacts first, then take succeeds (ADR 32, 34)', async () => {
    await world.move(LAMPLIGHT);
    npcSays(PIP, '*Pip lets go with a scowl.* "Keep it. It opens no doors without me."');
    const result = await play(
      "I snatch Pip's calling card.",
      react('Pip Quillon', 'The stranger snatches the calling card from your fingers.'),
      { toolCalls: [{ name: 'take', args: { item: 'calling card' } }] },
      (messages) => ({ text: lastToolText(messages).startsWith('The player now has calling card') ? 'Yours now.' : '?' }),
    );
    expect(result.reply).toBe('Yours now.');
    expect(result.events.map((e) => [e.kind, e.payload])).toEqual([
      ['npc.reacted', expect.objectContaining({ npc: PIP })],
      ['item.taken', { item: 'doc:calling-card', from: PIP }],
    ]);
  });

  it('snatching an item: take before the reaction is refused; a failed attempt takes nothing', async () => {
    await world.move(LAMPLIGHT);
    npcSays(PIP, '*Pip twists away and is gone into the crowd.*');
    const result = await play(
      "I snatch Pip's calling card.",
      {
        toolCalls: [
          { name: 'take', args: { item: 'calling card' } },
          { name: 'npc_react', args: { target: PIP, situation: 'The stranger grabs at your calling card.' } },
        ],
      },
      { text: 'Your fingers close on air.' },
    );
    const results = result.messages.filter((m) => m.getType() === 'tool').map((m) => m.text);
    expect(results[0]).toMatch(/^Error: Pip Quillon has calling card\. Play out the attempt first/);
    expect(result.events.map((e) => e.kind)).toEqual(['npc.reacted']);
    expect((await world.state.inventory(PIP)).map((e) => e.id)).toEqual(['doc:calling-card']);
  });

  it('refuses someone who is not here, and survives a failing NPC model', async () => {
    const result = await play(
      'I shout for Silas Ashcombe.',
      react('Silas Ashcombe', 'Someone shouts your name.'),
      react('Mags Holloway', 'The recruit shouts for Silas Ashcombe.'),
      { text: 'Nobody answers.' },
    );
    const results = result.messages.filter((m) => m.getType() === 'tool').map((m) => m.text);
    expect(results[0]).toBe('Error: there is no "Silas Ashcombe" here to react.');
    expect(results[1]).toMatch(/^Error: Mags Holloway could not react right now \(npc:mags-holloway was not expected/);
    expect(result.events).toEqual([]);
    expect(await chats.load('npc:mags-holloway')).toEqual([]);
  });
});

describe('NPC stream on its own channel', () => {
  it('sends the NPC request on channel npc:<id> with its name; the UI sees its deltas there', async () => {
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
        emit({ ...routed, type: 'delta', seq: 0, text: text.slice(0, 5) });
        emit({ ...routed, type: 'delta', seq: 1, text: text.slice(5) });
        const calls =
          request.purpose === 'gm' && requests.filter((r) => r.purpose === 'gm').length === 1
            ? [
                {
                  id: 'call_1',
                  type: 'function' as const,
                  function: { name: 'npc_react', arguments: JSON.stringify({ target: MAGS, situation: 'The recruit greets you.' }) },
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
    const agent = createGmAgent({
      ...models,
      world,
      meta,
      npc: { model: models.npcModel, chats, transcript },
    });
    const result = await runTurn({ agent, world, chats, transcript, input: 'Hello, Mags.' });

    const npcRequest = requests.find((r) => r.purpose === 'npc')!;
    expect(npcRequest).toMatchObject({ channel: MAGS, purpose: 'npc', meta: { speaker: 'Mags Holloway' }, sessionId: 'save-1' });
    expect(npcRequest.tools).toBeUndefined();
    expect(requests.filter((r) => r.purpose === 'gm').every((r) => r.channel === 'gm' && r.tools?.length)).toBe(true);
    const npcDeltas = seen.filter((e) => e.type === 'delta' && e.channel === MAGS);
    expect(npcDeltas.map((e) => (e as { text: string }).text).join('')).toBe('"Who\'s asking?"');
    expect(result.reply).toBe('Mags eyes you.');
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', MAGS, 'gm']);
  });
});
