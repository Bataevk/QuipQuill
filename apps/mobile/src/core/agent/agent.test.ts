import { AIMessage, type BaseMessage, ToolMessage } from '@langchain/core/messages';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { Db } from '@/core/storage/db';
import { readStoryMeta } from '@/core/story/story-meta';
import { openVectorIndex } from '@/core/vector/open-index';
import { toVector } from '@/core/vector/vector-index';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { ChatStore, GM_CHAT } from './chat-store';
import { createGmAgent, type GmAgent } from './gm-agent';
import { runTurn } from './run-turn';
import { TranscriptStore } from './transcript';
import { conversation, DEMO_TURNS, lastToolText, playDemoScenario, systemText } from './testing/demo-scenario';
import { ScriptedChatModel, type Step } from './testing/scripted-model';
import { GM_TOOL_NAMES } from './tools';

const TAVERN = 'loc:rusty-gear-tavern';
const LAMPLIGHT = 'loc:lamplight-row';
const PLAYER = 'player';
const CHARACTER = { id: 'c1', name: 'Aria', sex: 'female', age: 27, race: 'human', description: 'A cartographer.' };

let env: TestEnv;
let db: Db;
let world: World;
let chats: ChatStore;
let transcript: TranscriptStore;
let model: ScriptedChatModel;
let agent: GmAgent;
let mags: Float32Array;

beforeEach(async () => {
  env = await makeTestEnv();
  await env.library.importer.import(DEMO_STORY);
  await env.library.repository.addCharacter({ ...CHARACTER, createdAt: '2026-10-15T00:00:00.000Z' });
  const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: 'c1', title: 't' });
  db = await env.library.saves.open(save.id);
  world = await World.open(db, { vectors: await openVectorIndex(db) });
  const started = await world.startSession({ storyId: DEMO_STORY_ID, character: CHARACTER, startLocation: TAVERN });
  if (!started.ok) throw new Error(started.error);
  const row = await db.get<{ vector: Uint8Array }>(
    "SELECT vector FROM embeddings WHERE entity_id = 'npc:mags-holloway' AND kind = 'desc'",
  );
  mags = toVector(row!.vector);
  chats = new ChatStore(db);
  transcript = new TranscriptStore(db);
  model = new ScriptedChatModel([]);
  agent = createGmAgent({ model, world, meta: await readStoryMeta(db), embed: async () => mags });
});

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

/** Runs one turn with these model steps. */
async function turn(input: string, ...steps: Step[]) {
  model.push(...steps);
  const result = await runTurn({ agent, world, chats, transcript, input });
  expect(model.remaining).toBe(0);
  return result;
}

const toolTexts = (messages: BaseMessage[]) =>
  messages.filter((m) => ToolMessage.isInstance(m)).map((m) => m.text);

describe('demo scenario: look → go → take → inventory', () => {
  it('produces the right events, state and replies', async () => {
    const results = await playDemoScenario({ model, agent, world, chats, transcript });

    expect(results.map((r) => r.turn)).toEqual([1, 2, 3, 4]);
    expect(results.map((r) => r.events.map((e) => e.kind))).toEqual([[], ['player.moved'], ['item.taken'], []]);
    expect(results[1].events[0].payload).toEqual({ from: TAVERN, to: LAMPLIGHT });
    expect(results[2].events[0].payload).toEqual({ item: 'doc:discarded-evening-newspaper', from: LAMPLIGHT });
    expect(results.map((r) => r.reply)).toEqual([
      'Smoke curls under the low beams. Mags Holloway watches you over her mug.',
      'Gas lamps hiss along the wet cobbles of Lamplight Row.',
      'You fold the damp evening paper into your coat.',
      'Only the discarded evening newspaper, folded in your coat.',
    ]);

    const scene = await world.scene();
    expect(scene.location?.id).toBe(LAMPLIGHT);
    expect(scene.inventory.map((e) => e.id)).toEqual(['doc:discarded-evening-newspaper']);
    expect(model.boundTools).toEqual([...GM_TOOL_NAMES]);
  });

  it('keeps the history in model_messages and gives it back to the model next turn', async () => {
    await playDemoScenario({ model, agent, world, chats, transcript });
    const stored = await chats.load(GM_CHAT);
    // per turn: player input, AI (+ tool call), tool result, AI reply; the last turn has no tool
    expect(stored.map((m) => m.getType())).toEqual([
      ...['human', 'ai', 'tool', 'ai'],
      ...['human', 'ai', 'tool', 'ai'],
      ...['human', 'ai', 'tool', 'ai'],
      ...['human', 'ai'],
    ]);
    // the last model call got the whole history (all but its own reply) after the system prompt
    const lastCall = model.calls.at(-1)!;
    expect(conversation(lastCall).map((m) => m.text)).toEqual(stored.slice(0, -1).map((m) => m.text));
    expect((stored[1] as AIMessage).tool_calls?.[0]).toMatchObject({ name: 'inspect' });
    expect(DEMO_TURNS).toHaveLength(4);
  });
});

describe('system prompt', () => {
  it('caches system + history: the world state goes last and is rebuilt every call (ADR 46)', async () => {
    await turn(
      'I go upstairs.',
      { toolCalls: [{ name: 'set_relation', args: { a: 'player', relation: 'located_in', b: LAMPLIGHT } }] },
      { text: 'Up you go.' },
    );
    const [first, second] = model.calls;
    const parts = first[0].content as { type: string; text: string; cache_control?: unknown }[];
    expect(parts).toHaveLength(1);
    expect(parts[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(parts[0].text).toMatch(/# Story style\nTitle: The Brass Vault[\s\S]*steampunk/);
    expect(parts[0].text).not.toContain('{{');
    expect(parts[0].text).not.toContain('Location:');
    // the history ends with a breakpoint; the world state follows it
    const world1 = first.at(-1)!;
    expect(world1.text).toMatch(/^# World state\n[\s\S]*Player: Aria \(human, female, age 27\) \[player\]/);
    expect(world1.text).toContain('Location: The Rusty Gear Tavern [loc:rusty-gear-tavern]');
    const input = first.at(-2)!.content as { text: string; cache_control?: unknown }[];
    expect(input).toEqual([{ type: 'text', text: 'I go upstairs.', cache_control: { type: 'ephemeral' } }]);
    // the second call: the same system prompt, the same history as a prefix, the new world state last
    expect(second[0].content).toEqual(first[0].content);
    expect(second.slice(1, -3).map((m) => m.text)).toEqual(first.slice(1, -1).map((m) => m.text));
    const result = second.at(-2)!.content as { text: string; cache_control?: unknown }[];
    expect(result[0].cache_control).toEqual({ type: 'ephemeral' }); // on the tool result…
    expect((second.at(-4)!.content as { cache_control?: unknown }[])[0].cache_control).toEqual({ type: 'ephemeral' }); // …and the input
    expect(second.at(-1)!.text).toContain('Location: Lamplight Row [loc:lamplight-row]');
    expect(second.at(-1)!.text).toContain('The player went from The Rusty Gear Tavern to Lamplight Row.');
    // nothing of it is saved: the history has no breakpoints and no world state
    expect(JSON.stringify(await chats.load(GM_CHAT))).not.toMatch(/cache_control|# World state/);
  });

  it('lists exits, things here with what is inside them, inventory and notes', async () => {
    await db.run(
      // a story where the canister lies in the crate (the story's own graph, before the game reads it)
      "UPDATE relations SET dst = 'item:crate' WHERE src = 'item:smoke-canister' AND type = 'LOCATED_IN'",
    );
    await world.addNote('Mags wants the Clockheart before dawn.');
    await turn('Hm.', { text: 'Mm.' });
    const state = systemText(model.calls[0]);
    expect(state).toContain('Exits: Lamplight Row [loc:lamplight-row] — Up the stairs to the street → Lamplight Row.');
    expect(state).toMatch(/- crate \[item:crate\] \(item, fixed\): .* Inside: smoke canister \[item:smoke-canister\]\./);
    expect(state).toContain('Inventory: empty.');
    expect(state).toContain('- #1: Mags wants the Clockheart before dawn.');
  });
});

describe('tools', () => {
  it('returns broken rules as text and changes nothing', async () => {
    const result = await turn(
      'I walk into the vault.',
      { toolCalls: [{ name: 'set_relation', args: { a: 'player', relation: 'located_in', b: 'Orrery Vault' } }] },
      (messages) => ({ text: lastToolText(messages).startsWith('Error: There is no path') ? 'No way.' : '?' }),
    );
    expect(result.reply).toBe('No way.');
    expect(toolTexts(result.messages)[0]).toMatch(/Exits: Lamplight Row/);
    expect(result.events).toEqual([]);
    expect((await world.scene()).location?.id).toBe(TAVERN);
  });

  it('keeps the old rules through the general tools (ADR 43)', async () => {
    const rel = (a: string, relation: string, b: string) => ({ name: 'set_relation', args: { a, relation, b } });
    const refused = await turn(
      'I try everything at once.',
      { toolCalls: [rel('player', 'has_item', 'crate'), rel('player', 'located_in', 'Orrery Vault')] },
      { text: 'Nothing gives.' },
    );
    expect(toolTexts(refused.messages)).toEqual([
      'Error: crate is fixed in place and cannot be taken.',
      expect.stringMatching(/^Error: There is no path from The Rusty Gear Tavern to "Orrery Vault"\. Exits: Lamplight Row/),
    ]);
    expect(refused.events).toEqual([]);

    const moves = await turn(
      'I take the lantern, go up, leave the lantern there and try to grab Pip\'s card.',
      {
        toolCalls: [
          rel('player', 'has_item', 'brass lantern'),
          rel('player', 'located_in', 'Lamplight Row'),
          rel('brass lantern', 'located_in', 'Lamplight Row'),
          rel('player', 'has_item', 'calling card'),
        ],
      },
      { text: 'Pip keeps his card.' },
    );
    expect(toolTexts(moves.messages)[3]).toMatch(/^Error: Pip Quillon has calling card\. Play out the attempt first/);
    expect(moves.events.map((e) => [e.kind, e.payload])).toEqual([
      ['item.taken', { item: 'item:brass-lantern', from: TAVERN }],
      ['player.moved', { from: TAVERN, to: LAMPLIGHT }],
      ['item.dropped', { item: 'item:brass-lantern', to: LAMPLIGHT }],
    ]);
  });

  it('runs several calls of one reply in the order the model wrote them', async () => {
    const result = await turn(
      'I go up and grab the newspaper.',
      {
        toolCalls: [
          { name: 'set_relation', args: { a: 'player', relation: 'located_in', b: 'Lamplight Row' } },
          { name: 'set_relation', args: { a: 'player', relation: 'has_item', b: 'newspaper' } },
        ],
      },
      { text: 'Done.' },
    );
    expect(result.events.map((e) => e.kind)).toEqual(['player.moved', 'item.taken']);
  });

  it('inspects people, containers and the player', async () => {
    const result = await turn(
      'I study Mags and the crate.',
      {
        toolCalls: [
          { name: 'inspect', args: { target: 'Mags Holloway' } },
          { name: 'inspect', args: { target: 'crate' } },
          { name: 'inspect', args: { target: 'Aria' } },
          { name: 'inspect', args: { target: 'loc:orrery-vault' } },
          { name: 'inspect', args: { target: 'Orrery Vault' } },
        ],
      },
      { text: 'You look.' },
    );
    const [person, crate, player, byId, byName] = toolTexts(result.messages);
    expect(person).toMatch(/^Mags Holloway \[npc:mags-holloway\] \(npc\)\n.*captain/);
    expect(person).toContain('Where: The Rusty Gear Tavern [loc:rusty-gear-tavern].');
    expect(crate).toContain('(item, fixed in place)');
    expect(player).toMatch(/^Aria \[player\] \(player\)\nA cartographer\./);
    // out of sight: only by an id the master already knows (from lore search or events)
    expect(byId).toMatch(/^Orrery Vault \[loc:orrery-vault\] \(location\)\n[^]*\nExits: Old Sewer Culvert/);
    expect(byName).toBe('Error: there is no "Orrery Vault" here or in the inventory.');
  });

  it("refuses an NPC's item until the NPC reacted (ADR 32, 34)", async () => {
    await world.setRelation('player', 'LOCATED_IN', LAMPLIGHT);
    const result = await turn(
      "I snatch Pip's calling card.",
      { toolCalls: [{ name: 'set_relation', args: { a: 'player', relation: 'has_item', b: 'calling card' } }] },
      { text: 'Pip pulls it back.' },
    );
    expect(toolTexts(result.messages)[0]).toMatch(/^Error: Pip Quillon has calling card\. Play out the attempt/);
    expect(result.events).toEqual([]);
  });

  it('creates, edits, updates the player, takes notes, drops and searches the lore', async () => {
    const result = await turn(
      'I rig a torch, scare Mags, tear my coat and drop the torch.',
      {
        toolCalls: [
          { name: 'set_entity', args: { name: 'makeshift torch', type: 'item', description: 'Rags on a stick.' } },
          { name: 'set_relation', args: { a: 'player', relation: 'has_item', b: 'makeshift torch' } },
          { name: 'set_entity', args: { id: 'Mags Holloway', description: 'Mags, now wary of fire.' } },
          { name: 'set_entity', args: { id: 'player', description: 'A cartographer in a torn coat.' } },
          { name: 'note', args: { text: 'Mags owes the player a drink.' } },
          { name: 'set_relation', args: { a: 'makeshift torch', relation: 'located_in', b: 'The Rusty Gear Tavern' } },
          { name: 'note', args: {} },
          { name: 'search_lore', args: { query: 'airship captain' } },
        ],
      },
      { text: 'Smoke and apologies.' },
    );
    const texts = toolTexts(result.messages);
    expect(texts.slice(0, 7)).toEqual([
      'Created makeshift torch [item:makeshift-torch] here. It is now a lasting part of the world.',
      'The player now has makeshift torch [item:makeshift-torch] (taken from The Rusty Gear Tavern).',
      'Mags Holloway [npc:mags-holloway] is updated.',
      'Aria [player] is updated.',
      'Note #1 is added.',
      'makeshift torch [item:makeshift-torch] now lies in The Rusty Gear Tavern.',
      'Error: give the text of a new note, or close_id to close one.',
    ]);
    // the query is her story description, which set_entity has just rewritten: the lore no
    // longer finds her by it (ADR 42); the rest of the lore answers
    expect(texts[7].split('\n')[0]).toMatch(/^- .+ \[[a-z]+:[a-z-]+\] \([a-z]+\): /);
    expect(texts[7]).not.toContain('[npc:mags-holloway]');
    expect(result.events.map((e) => e.kind)).toEqual([
      'entity.added',
      'item.taken',
      'entity.edited',
      'entity.edited',
      'note.added',
      'item.dropped',
    ]);
    expect((await world.state.entity(PLAYER))?.description).toBe('A cartographer in a torn coat.');
  });

  it('answers without lore search when there are no embeddings', async () => {
    const offline = createGmAgent({ model, world, meta: await readStoryMeta(db) });
    model.push({ toolCalls: [{ name: 'search_lore', args: { query: 'x' } }] }, { text: 'ok' });
    const result = await runTurn({ agent: offline, world, chats, transcript, input: 'Who is Mags?' });
    expect(toolTexts(result.messages)).toEqual(['Error: lore search is not available right now. Rely on the world state.']);
  });

  it('turns a thrown failure into text', async () => {
    const broken = createGmAgent({
      model,
      world,
      meta: await readStoryMeta(db),
      embed: async () => {
        throw new Error('network down');
      },
    });
    model.push({ toolCalls: [{ name: 'search_lore', args: { query: 'x' } }] }, { text: 'ok' });
    const result = await runTurn({ agent: broken, world, chats, transcript, input: 'Who is Mags?' });
    expect(toolTexts(result.messages)[0]).toBe(
      'Error: search_lore failed unexpectedly (network down). Narrate around it; do not retry.',
    );
  });

  it('answers an unknown tool with an error the model sees', async () => {
    const result = await turn(
      'I fly.',
      { toolCalls: [{ name: 'fly', args: {} }] },
      (messages) => ({ text: /fly/.test(lastToolText(messages)) ? 'You flap your arms.' : '?' }),
    );
    expect(result.reply).toBe('You flap your arms.');
  });
});

describe('invalid tool call arguments', () => {
  it('reports them to the model as tool errors instead of ending the turn', async () => {
    const result = await turn(
      'I go upstairs.',
      { invalidToolCalls: [{ name: 'set_relation', args: '{"to": "Lamplight' }] },
      (messages) => {
        expect(lastToolText(messages)).toMatch(/^Error: the arguments of set_relation call are not a valid JSON object/);
        return { toolCalls: [{ name: 'set_relation', args: { a: 'player', relation: 'located_in', b: 'Lamplight Row' } }] };
      },
      { text: 'Up the stairs.' },
    );
    expect(result.reply).toBe('Up the stairs.');
    expect(result.events.map((e) => e.kind)).toEqual(['player.moved']);
    // The broken call stays in the history as a regular call with its raw arguments, answered.
    const ai = result.messages[1] as AIMessage;
    expect(ai.tool_calls).toEqual([
      { id: 'bad_1_0', name: 'set_relation', args: { invalid_arguments: '{"to": "Lamplight' }, type: 'tool_call' },
    ]);
    expect(ai.invalid_tool_calls).toEqual([]);
    expect((result.messages[2] as ToolMessage).tool_call_id).toBe('bad_1_0');
  });

  it('still runs the valid calls of the same reply', async () => {
    const result = await turn(
      'I go up and grab the paper.',
      {
        toolCalls: [{ name: 'set_relation', args: { a: 'player', relation: 'located_in', b: 'Lamplight Row' } }],
        invalidToolCalls: [{ name: 'set_relation', args: '{item' }],
      },
      (messages) => {
        const tools = messages.filter((m) => ToolMessage.isInstance(m)).map((m) => m.text);
        expect(tools).toHaveLength(2);
        return { toolCalls: [{ name: 'set_relation', args: { a: 'player', relation: 'has_item', b: 'newspaper' } }] };
      },
      { text: 'Got it.' },
    );
    expect(result.events.map((e) => e.kind)).toEqual(['player.moved', 'item.taken']);
    const stored = await chats.load(GM_CHAT);
    const calls = stored.flatMap((m) => (AIMessage.isInstance(m) ? (m.tool_calls ?? []) : []));
    const answered = new Set(stored.flatMap((m) => (ToolMessage.isInstance(m) ? [m.tool_call_id] : [])));
    expect(calls.every((call) => answered.has(call.id!))).toBe(true);
  });
});

describe('roll_luck (ADR 47)', () => {
  const roll = (likelihood: string, stakes: string) => ({
    toolCalls: [{ name: 'roll_luck', args: { attempt: 'search under the tables', likelihood, stakes } }],
  });
  const play = async (luck: number, draws: number[], ...steps: Step[]) => {
    model.push(...steps);
    const meta = await readStoryMeta(db);
    expect(meta.luck).toBe(0.5); // the demo story does not say: a fair world
    const lucky = createGmAgent({ model, world, meta: { ...meta, luck }, random: () => draws.shift() ?? 0.5 });
    return runTurn({ agent: lucky, world, chats, transcript, input: 'I search under the tables.' });
  };

  it('rolls in code with the story luck; the master hears the outcome and decides what it is', async () => {
    const result = await play(0.5, [0.2], roll('possible', 'trivial'), (messages) => {
      expect(lastToolText(messages)).toMatch(/^Luck: it works out\. Narrate what the player finds or gains: .*trivial at most/);
      return { text: 'A tarnished penny, stuck to the floor.' };
    });
    expect(result.events.map((e) => [e.kind, e.payload])).toEqual([
      [
        'luck.rolled',
        { attempt: 'search under the tables', likelihood: 'possible', stakes: 'trivial', luck: 0.5, chance: 0.45, roll: 0.2, outcome: 'success' },
      ],
    ]);
    // the master does not create it: the generator records what he narrated, after the reply
    expect((await world.scene()).inventory).toEqual([]);
  });

  it('fails plainly in a fair world, and turns into a mishap in a very unlucky one', async () => {
    await play(0.5, [0.99, 0], roll('possible', 'useful'), (messages) => {
      expect(lastToolText(messages)).toBe('Luck: it does not work out. Nothing is found or gained; narrate that plainly.');
      return { text: 'Only dust.' };
    });
    await play(0.05, [0.99, 0.1], roll('possible', 'useful'), (messages) => {
      expect(lastToolText(messages)).toMatch(/^Luck: bad luck\. It fails, and something small goes wrong/);
      return { text: 'You knock over a mug; heads turn.' };
    });
    expect((await world.log.recent(2)).map((e) => e.kind === 'luck.rolled' && e.payload.outcome)).toEqual(['failure', 'mishap']);
  });
});
