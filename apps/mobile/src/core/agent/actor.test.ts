import type { AIMessage, BaseMessage } from '@langchain/core/messages';
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
/** Actor answers by entity id; every call is recorded with who was asked. */
let actorScripts: Map<string, ScriptedChatModel>;
let actorCalls: { entity: Entity; messages: BaseMessage[] }[];
/** Lines of the actor pack by entity id (default: a nod); every call is recorded. */
let lineScripts: Map<string, Step[]>;
let lineCalls: { entity: Entity; messages: BaseMessage[]; tools: string[] }[];

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
  actorScripts = new Map();
  actorCalls = [];
  lineScripts = new Map();
  lineCalls = [];
});

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

/** The entity with this id will answer these lines, in order. */
function feels(id: string, ...lines: string[]) {
  const model = actorScripts.get(id) ?? new ScriptedChatModel([]);
  model.push(...lines.map((text) => ({ text })));
  actorScripts.set(id, model);
}

const actorModel = (entity: Entity) => {
  const scripted = actorScripts.get(entity.id);
  if (!scripted) throw new Error(`${entity.id} was not expected to react`);
  return {
    invoke: async (messages: BaseMessage[]) => {
      actorCalls.push({ entity, messages });
      return scripted.invoke(messages);
    },
  } as unknown as ScriptedChatModel;
};

/** The entity with this id will write these lines in the actor pack, in order. */
function says(id: string, ...lines: Step[]) {
  lineScripts.set(id, [...(lineScripts.get(id) ?? []), ...lines]);
}

const lineModel = (entity: Entity) => {
  const model = new ScriptedChatModel([]);
  const next = lineScripts.get(entity.id)?.shift() ?? { text: `*${entity.name} nods.*` };
  model.push((messages) => {
    lineCalls.push({ entity, messages, tools: model.boundTools });
    return typeof next === 'function' ? next(messages) : next;
  });
  return model;
};

function agentFor() {
  return createGmAgent({ model: master, world, meta, actor: { model: actorModel, lineModel } });
}

function play(input: string, ...steps: Step[]) {
  master.push(...steps);
  return runTurn({ agent: agentFor(), world, chats, transcript, input });
}

const react = (target: string, situation: string) => ({ toolCalls: [{ name: 'react_entity', args: { target, situation } }] });
const call = (name: string, args: Record<string, unknown>) => ({ name, args });
const promptOf = (i: number) => actorCalls[i].messages[0].content as { text: string; cache_control?: unknown }[];

describe('react_entity', () => {
  it('full cycle: react → her line in the actor pack → actor_note → the master answers (ADR 43)', async () => {
    const voice = 'A rookie asking about the vault, out loud? I will tell them it is below the bank, nothing more.';
    feels(MAGS, voice);
    says(MAGS, (messages) => {
      expect(lastToolText(messages)).toMatch(/^Inner voice of Mags Holloway \[npc:mags-holloway\] — for you only/);
      expect(messages.at(-1)!.text).toMatch(/^\[Actor\] Now speak and act as Mags Holloway \[npc:mags-holloway\]/);
      expect(messages.at(-1)!.text).toContain('a brass hook');
      return { text: '*She taps the plan.* "Below the bank, love. That\'s all you get."', toolCalls: [call('note', { text: 'x' })] };
    });
    const result = await play(
      'I ask Mags about the vault.',
      react('Mags Holloway', 'The new recruit asks you about the vault.'),
      (messages) => {
        expect(master.boundTools).toContain('actor_note');
        expect(messages.at(-2)!.text).toBe('Mags Holloway [npc:mags-holloway]: *She taps the plan.* "Below the bank, love. That\'s all you get."');
        expect(messages.at(-1)!.text).toMatch(/^\[Game Master\] Mags Holloway had their say; the player saw it\./);
        return { toolCalls: [call('actor_note', { text: 'I told the rookie only "below the bank"; I do not trust them yet.' })] };
      },
      (messages) => {
        expect(lastToolText(messages)).toBe('Noted for Mags Holloway.');
        // the instruction stays right after her line: the master still owes the player an answer
        expect(messages.at(-1)!.getType()).toBe('tool');
        expect(messages.at(-3)!.text).toMatch(/^\[Game Master\] Mags Holloway had their say/);
        expect(messages.at(-4)!.text).toMatch(/^Mags Holloway \[npc:mags-holloway\]: \*She taps/);
        expect(master.boundTools).toContain('actor_note');
        return { text: 'The tavern noise swallows the rest.' };
      },
    );
    expect(result.reply).toBe('The tavern noise swallows the rest.');
    expect(result.events.map((e) => [e.kind, e.payload])).toEqual([
      ['entity.reacted', { entity: MAGS, situation: 'The new recruit asks you about the vault.', reaction: voice }],
      ['note.added', { id: 1, text: 'I told the rookie only "below the bank"; I do not trust them yet.', entity: MAGS }],
    ]);
    // the line: its own bubble, from her; a tool call in the actor pack is dropped
    expect(lineCalls.map((c) => c.entity.id)).toEqual([MAGS]);
    expect((await transcript.lines()).map((l) => [l.speaker, l.text.slice(0, 18), l.extra])).toEqual([
      ['player', 'I ask Mags about t', {}],
      [MAGS, '*She taps the plan', { name: 'Mags Holloway' }],
      ['gm', 'The tavern noise s', {}],
    ]);
    const line = (await chats.load(GM_CHAT)).find((m) => m.additional_kwargs?.qq_actor);
    expect(line?.text).toBe('*She taps the plan.* "Below the bank, love. That\'s all you get."');
    expect((line as AIMessage).tool_calls).toEqual([]);
    expect((await world.scene()).notes).toEqual([]);

    // the inner voice prompt
    const [stable, scene] = promptOf(0);
    expect(stable.cache_control).toEqual({ type: 'ephemeral' });
    expect(stable.text).toMatch(/^You are Mags Holloway, in an interactive story\./);
    expect(stable.text).toContain('- Silas Ashcombe: revenge on Silas Ashcombe, who seized her airship');
    expect(stable.text).toContain('answer them with your inner voice');
    expect(scene.text).toContain('The player character: Aria (human) — A cartographer.');

    // the next turn: the standard pack; her note reaches her next inner voice
    feels(MAGS, 'Them again.');
    await play(
      'Anything else?',
      () => {
        expect(master.boundTools).not.toContain('actor_note');
        return react(MAGS, 'The rookie asks for more.');
      },
      { text: 'She shrugs.' },
    );
    expect(promptOf(1)[1].text).toContain('# Your notes (what you decided before)\n- I told the rookie only');
  });

  it('a magic mirror in the inventory reacts too', async () => {
    feels('item:magic-mirror', 'At last someone looks at me. I want them to ask about the vault — I saw its keeper.');
    const result = await play(
      'I pull out the mirror I found and look into it.',
      {
        toolCalls: [
          call('set_entity', { name: 'magic mirror', type: 'item', description: 'A hand mirror that remembers every face it has shown.' }),
        ],
      },
      { toolCalls: [call('set_relation', { a: 'player', relation: 'has_item', b: 'magic mirror' })] },
      react('magic mirror', 'The stranger who carries you looks into you.'),
      (messages) => ({ text: lastToolText(messages).startsWith('Inner voice of magic mirror') ? 'The glass ripples.' : '?' }),
    );
    expect(result.reply).toBe('The glass ripples.');
    expect(result.events.map((e) => e.kind)).toEqual(['entity.added', 'item.taken', 'entity.reacted']);
    const [stable, scene] = promptOf(0);
    expect(stable.text).toContain('remembers every face');
    expect(scene.text).toContain('The player character carries you.');
    expect(scene.text).toContain('You are in The Rusty Gear Tavern');
  });

  it('a stone stays silent: [silence] reaches the master as "does not react"', async () => {
    feels('item:crate', SILENCE);
    const result = await play('I ask the crate where the gold is.', react('crate', 'Someone asks you where the gold is.'), (messages) => {
      expect(lastToolText(messages)).toBe('crate [item:crate] does not react.');
      return { text: 'The crate keeps its counsel.' };
    });
    expect(result.events.map((e) => e.payload)).toEqual([
      { entity: 'item:crate', situation: 'Someone asks you where the gold is.', reaction: SILENCE },
    ]);
    expect(lineCalls).toEqual([]); // no line for silence
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', 'gm']);
  });

  it('remembers through actor notes and its last reactions, not a chat', async () => {
    feels(MAGS, 'Who is this? I want to know before I say anything.', 'Them again. Still no.');
    await play('Hello, Mags.', react(MAGS, 'A stranger greets you.'), { text: '.' });
    expectOk(await world.addActorNote(MAGS, 'I refused to tell the stranger about the vault.'));
    await play('Tell me about the vault.', react(MAGS, 'The stranger asks about the vault.'), { text: '.' });

    expect(promptOf(0)[1].text).not.toContain('# Your notes');
    const scene = promptOf(1)[1].text;
    expect(scene).toContain('# Your notes (what you decided before)\n- I refused to tell the stranger about the vault.');
    expect(scene).toContain('# Lately\n- A stranger greets you. → you felt: Who is this?');
    expect(actorCalls[1].messages.map((m) => m.getType())).toEqual(['system', 'human']);
    // the notes are the actor's, not the master's
    expect((await world.scene()).notes).toEqual([]);
  });

  it('asks an innkeeper the master just added to the scene', async () => {
    feels('npc:innkeeper', 'Another one who wants talk, not ale. Talk costs extra.');
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
    const [stable, scene] = promptOf(0);
    expect(stable.text).toContain('A bald man polishing mugs');
    expect(stable.text).not.toContain('# Your ties');
    expect(scene.text).toContain('Also here: Mags Holloway.');
    expect(scene.text).not.toContain('You carry');
  });

  it('snatching an item: the owner reacts first, then take succeeds (ADR 32, 34)', async () => {
    await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT);
    feels(PIP, 'Fine, let it go — the card opens no doors without me anyway.');
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

  it('snatching an item: take before the reaction is refused; a failed attempt takes nothing', async () => {
    await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT);
    feels(PIP, 'Not a chance. I twist away into the crowd.');
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

  it('returns an error when the reaction cannot be journaled', async () => {
    feels(MAGS, 'Hm.');
    vi.spyOn(world, 'recordReaction').mockResolvedValue({ ok: false, error: 'disk full' });
    const result = await play('Hi, Mags.', react(MAGS, 'The recruit says hi.'), { text: 'Silence.' });
    expect(result.messages.find((m) => m.getType() === 'tool')?.text).toMatch(/^Error: Mags Holloway could not react/);
  });

  it('refuses something that is not here, and survives a failing actor model', async () => {
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
  });
});

describe('packs (ADR 43)', () => {
  const retry = () => retryTurn({ agent: agentFor(), world, chats, transcript });

  it('a turn cut in the actor pack is retried from the line', async () => {
    feels(MAGS, 'I want them gone.');
    says(MAGS, () => {
      throw new Error('network');
    });
    await expect(play('Hi, Mags.', react(MAGS, 'The recruit says hi.'))).rejects.toThrow('network');
    expect(isUnfinished(await chats.load(GM_CHAT))).toBe(true);

    says(MAGS, { text: '"Shove off."' });
    master.push((messages) => ({ text: messages.at(-1)!.text.startsWith('[Game Master]') ? 'She turns away.' : '?' }));
    const result = await retry();
    expect(result.reply).toBe('She turns away.');
    expect(actorCalls).toHaveLength(1); // the inner voice is not asked again
    expect(result.events.map((e) => e.kind)).toEqual(['entity.reacted']);
    expect((await transcript.lines()).map((l) => [l.speaker, l.text])).toEqual([
      ['player', 'Hi, Mags.'],
      [MAGS, '"Shove off."'],
      ['gm', 'She turns away.'],
    ]);
  });

  it('a turn cut after the line goes on in the master pack, the line is not repeated', async () => {
    feels(MAGS, 'I want them gone.');
    says(MAGS, { text: '"Shove off."' });
    await expect(
      play('Hi, Mags.', react(MAGS, 'The recruit says hi.'), () => {
        throw new Error('network');
      }),
    ).rejects.toThrow('network');
    const saved = await chats.load(GM_CHAT);
    expect(saved.at(-1)!.additional_kwargs?.qq_actor).toEqual({ id: MAGS, name: 'Mags Holloway' });
    expect(isUnfinished(saved)).toBe(true);

    master.push(
      (messages) => {
        expect(messages.at(-1)!.text).toMatch(/^\[Game Master\] Mags Holloway had their say/);
        return { toolCalls: [call('actor_note', { text: 'I sent the recruit away.' })] };
      },
      { text: 'She turns away.' },
    );
    const result = await retry();
    expect(lineCalls).toHaveLength(1);
    expect(result.events.map((e) => e.kind)).toEqual(['entity.reacted', 'note.added']);
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', MAGS, 'gm']);
  });

  it('"the door in the alley": what an honest NPC says exists when the player gets there', async () => {
    feels(MAGS, 'The recruit needs the way in. I will tell the truth: the service door on Lamplight Row, behind the barrels.');
    says(MAGS, { text: '"Lamplight Row. Service door behind the barrels. Knock twice."' });
    await play(
      'How do I get into the bank unseen?',
      react(MAGS, 'The recruit asks how to get into the bank unseen.'),
      (messages) => {
        expect(lastToolText(messages)).toContain('I will tell the truth');
        return {
          toolCalls: [
            call('set_entity', { name: 'service door', type: 'other', description: 'A narrow door behind the barrels; two knocks open it.' }),
            call('set_relation', { a: 'service door', relation: 'located_in', b: 'Lamplight Row' }),
            call('actor_note', { text: 'I showed the recruit the service door.' }),
          ],
        };
      },
      { text: 'She goes back to her ale.' },
    );
    await play(
      'I go to Lamplight Row and look for the door.',
      { toolCalls: [call('set_relation', { a: 'player', relation: 'located_in', b: 'Lamplight Row' })] },
      { toolCalls: [call('inspect', { target: 'service door' })] },
      (messages) => ({ text: lastToolText(messages).includes('two knocks open it') ? 'There it is.' : 'Nothing.' }),
    );
    expect((await world.scene()).here.map((e) => e.id)).toContain('ent:service-door');
    expect(master.calls.at(-1)!.at(-1)!.text).toContain('two knocks open it');
  });

  it('two react in one reply: each speaks in turn, then notes go to whom they belong', async () => {
    feels(MAGS, 'Another mouth to feed. No.');
    feels('npc:innkeeper', 'Trouble at my table. I want them out.');
    says(MAGS, { text: '"No."' });
    says('npc:innkeeper', { text: '*He wipes the table, pointedly.*' });
    const result = await play(
      'I ask Mags and the innkeeper for a room.',
      {
        toolCalls: [
          call('set_entity', { name: 'innkeeper', type: 'npc', description: 'A bald man polishing mugs.' }),
          call('react_entity', { target: MAGS, situation: 'The stranger asks you for a room.' }),
          call('react_entity', { target: 'innkeeper', situation: 'The stranger asks for a room.' }),
        ],
      },
      (messages) => {
        expect(messages.at(-1)!.text).toMatch(/^\[Game Master\] Mags Holloway and innkeeper had their say/);
        return {
          toolCalls: [
            call('actor_note', { text: 'I refused the stranger.', who: 'Mags' }),
            call('actor_note', { text: 'I want the stranger out.' }),
            call('actor_note', { text: 'x', who: 'Silas Ashcombe' }),
          ],
        };
      },
      { text: 'No room tonight.' },
    );
    expect(lineCalls.map((c) => c.entity.id)).toEqual([MAGS, 'npc:innkeeper']);
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['player', MAGS, 'npc:innkeeper', 'gm']);
    expect(result.messages.filter((m) => m.getType() === 'tool').slice(-1)[0].text).toMatch(
      /^Error: "Silas Ashcombe" did not react this turn\. Choose one of: Mags Holloway, innkeeper\./,
    );
    expect((await world.log.actorNotes(MAGS, 5)).map((n) => n.text)).toEqual(['I refused the stranger.']);
    expect((await world.log.actorNotes('npc:innkeeper', 5)).map((n) => n.text)).toEqual(['I want the stranger out.']);
  });
});

describe('actor streams', () => {
  it('the inner voice goes on the hidden channel "actor", the line on npc:<id> with her name', async () => {
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
        const text = request.purpose === 'npc' ? 'Who is asking? I do not like it.' : 'Mags eyes you.';
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
    const agent = createGmAgent({
      ...models,
      world,
      meta,
      actor: { model: () => models.actorModel, lineModel: models.actorLineModel },
    });
    const result = await runTurn({ agent, world, chats, transcript, input: 'Hello, Mags.' });

    const [voice, line] = requests.filter((r) => r.purpose === 'npc');
    expect(voice).toMatchObject({ channel: 'actor', purpose: 'npc', sessionId: 'save-1' });
    expect(voice.meta).toBeUndefined();
    expect(voice.tools).toBeUndefined();
    // the line streams in her bubble; the master's tools stay (cache, tool history)
    expect(line).toMatchObject({ channel: MAGS, purpose: 'npc', meta: { speaker: 'Mags Holloway' } });
    expect(line.tools?.map((t) => t.function.name)).not.toContain('actor_note');
    const gm = requests.filter((r) => r.purpose === 'gm');
    expect(gm[1].tools?.map((t) => t.function.name)).toContain('actor_note');
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
  it('what an honest NPC tells about becomes real: the master creates the bag, the player takes it (ADR 43)', async () => {
    feels(MAGS, '"Tools? Under the bench by the stairs. A canvas bag, mind the chisels."');
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
    feels(MAGS, '"Took your time."');
    await play('Hello again.', react(MAGS, 'The stranger catches up with you.'), { text: '.' });
    expect((actorCalls[0].messages[0].content as { text: string }[])[1].text).toContain('You are in Lamplight Row');
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

    feels(MAGS, '"Walk away, then."');
    await play('I nod to Mags.', react(MAGS, 'The stranger who turned you down nods to you.'), { text: '.' });
    const stable = (actorCalls[0].messages[0].content as { text: string }[])[0].text;
    expect(stable).toContain('- Aria: the stranger refused to join the heist and took Pip; she no longer trusts them');
    expect(stable).not.toContain('leader of the crew');
    expect(stable).not.toContain('recruited the player');
    expect(stable).toContain('Her crew fell apart; she plans alone now.');
  });
});
