import { AIMessage, type BaseMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
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
import { DEMO_TURNS, lastToolText, playDemoScenario, systemText } from './testing/demo-scenario';
import { ScriptedChatModel, type Step } from './testing/scripted-model';
import { GM_TOOL_NAMES } from './tools';

const TAVERN = 'loc:rusty-gear-tavern';
const LAMPLIGHT = 'loc:lamplight-row';
const PLAYER = 'player:c1';
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
    expect(lastCall.slice(1).map((m) => m.text)).toEqual(stored.slice(0, -1).map((m) => m.text));
    expect((stored[1] as AIMessage).tool_calls?.[0]).toMatchObject({ name: 'inspect' });
    expect(DEMO_TURNS).toHaveLength(4);
  });
});

describe('system prompt', () => {
  it('puts the cache breakpoint after the story style and rebuilds the world state every call', async () => {
    await turn(
      'I go upstairs.',
      { toolCalls: [{ name: 'move', args: { to: LAMPLIGHT } }] },
      { text: 'Up you go.' },
    );
    const [first, second] = model.calls.map((call) => call[0] as SystemMessage);
    const parts = first.content as { type: string; text: string; cache_control?: unknown }[];
    expect(parts).toHaveLength(2);
    expect(parts[0].cache_control).toEqual({ type: 'ephemeral' });
    expect(parts[0].text).toContain('# Story style\nTitle: The Brass Vault');
    expect(parts[0].text).toMatch(/steampunk/);
    expect(parts[0].text).not.toContain('{{');
    expect(parts[1].text).toMatch(/^Player: Aria \(human, female, age 27\) \[player:c1\]/);
    expect(parts[1].text).toContain('Location: The Rusty Gear Tavern [loc:rusty-gear-tavern]');
    // the second call sees the result of the move, with the same cached prefix
    const secondParts = second.content as { text: string }[];
    expect(secondParts[0].text).toBe(parts[0].text);
    expect(secondParts[1].text).toContain('Location: Lamplight Row [loc:lamplight-row]');
    expect(secondParts[1].text).toContain('The player went from The Rusty Gear Tavern to Lamplight Row.');
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
      { toolCalls: [{ name: 'move', args: { to: 'Orrery Vault' } }] },
      (messages) => ({ text: lastToolText(messages).startsWith('Error: There is no path') ? 'No way.' : '?' }),
    );
    expect(result.reply).toBe('No way.');
    expect(toolTexts(result.messages)[0]).toMatch(/Exits: Lamplight Row/);
    expect(result.events).toEqual([]);
    expect((await world.scene()).location?.id).toBe(TAVERN);
  });

  it('runs several calls of one reply in the order the model wrote them', async () => {
    const result = await turn(
      'I go up and grab the newspaper.',
      {
        toolCalls: [
          { name: 'move', args: { to: 'Lamplight Row' } },
          { name: 'take', args: { item: 'newspaper' } },
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
    expect(player).toMatch(/^Aria \[player:c1\] \(player\)\nA cartographer\./);
    // out of sight: only by an id the master already knows (from lore search or events)
    expect(byId).toMatch(/^Orrery Vault \[loc:orrery-vault\] \(location\)\n[^]*\nExits: Old Sewer Culvert/);
    expect(byName).toBe('Error: there is no "Orrery Vault" here or in the inventory.');
  });

  it("refuses an NPC's item until the NPC reacted (ADR 32, 34)", async () => {
    await world.move(LAMPLIGHT);
    const result = await turn(
      "I snatch Pip's calling card.",
      { toolCalls: [{ name: 'take', args: { item: 'calling card' } }] },
      { text: 'Pip pulls it back.' },
    );
    expect(toolTexts(result.messages)[0]).toMatch(/^Error: Pip Quillon has calling card\. Play out the attempt/);
    expect(result.events).toEqual([]);
  });

  it('adds, edits, updates the player, takes notes, drops and searches the lore', async () => {
    const result = await turn(
      'I rig a torch, scare Mags, tear my coat and drop the torch.',
      {
        toolCalls: [
          { name: 'add_entity', args: { name: 'makeshift torch', type: 'item', description: 'Rags on a stick.', place: 'inventory' } },
          { name: 'edit_entity', args: { target: 'Mags Holloway', description: 'Mags, now wary of fire.' } },
          { name: 'update_player', args: { description: 'A cartographer in a torn coat.' } },
          { name: 'note', args: { text: 'Mags owes the player a drink.' } },
          { name: 'drop', args: { item: 'makeshift torch' } },
          { name: 'note', args: { close_id: 1 } },
          { name: 'note', args: {} },
          { name: 'search_lore', args: { query: 'airship captain' } },
        ],
      },
      { text: 'Smoke and apologies.' },
    );
    const texts = toolTexts(result.messages);
    expect(texts.slice(0, 7)).toEqual([
      'Added makeshift torch [item:makeshift-torch] to the inventory. It is now a lasting part of the world.',
      'Mags Holloway [npc:mags-holloway] is updated.',
      "The player's description is updated.",
      'Note #1 is added.',
      'makeshift torch [item:makeshift-torch] now lies in The Rusty Gear Tavern.',
      'Note #1 is closed.',
      'Error: give the text of a new note, or close_id to close one.',
    ]);
    expect(texts[7].split('\n')[0]).toMatch(/^- Mags Holloway \[npc:mags-holloway\] \(npc\): /);
    expect(result.events.map((e) => e.kind)).toEqual([
      'entity.added',
      'entity.edited',
      'player.updated',
      'note.added',
      'item.dropped',
      'note.closed',
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
      { invalidToolCalls: [{ name: 'move', args: '{"to": "Lamplight' }] },
      (messages) => {
        expect(lastToolText(messages)).toMatch(/^Error: the arguments of move call are not a valid JSON object/);
        return { toolCalls: [{ name: 'move', args: { to: 'Lamplight Row' } }] };
      },
      { text: 'Up the stairs.' },
    );
    expect(result.reply).toBe('Up the stairs.');
    expect(result.events.map((e) => e.kind)).toEqual(['player.moved']);
    // The broken call stays in the history as a regular call with its raw arguments, answered.
    const ai = result.messages[1] as AIMessage;
    expect(ai.tool_calls).toEqual([
      { id: 'bad_1_0', name: 'move', args: { invalid_arguments: '{"to": "Lamplight' }, type: 'tool_call' },
    ]);
    expect(ai.invalid_tool_calls).toEqual([]);
    expect((result.messages[2] as ToolMessage).tool_call_id).toBe('bad_1_0');
  });

  it('still runs the valid calls of the same reply', async () => {
    const result = await turn(
      'I go up and grab the paper.',
      {
        toolCalls: [{ name: 'move', args: { to: 'Lamplight Row' } }],
        invalidToolCalls: [{ name: 'take', args: '{item' }],
      },
      (messages) => {
        const tools = messages.filter((m) => ToolMessage.isInstance(m)).map((m) => m.text);
        expect(tools).toHaveLength(2);
        return { toolCalls: [{ name: 'take', args: { item: 'newspaper' } }] };
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
