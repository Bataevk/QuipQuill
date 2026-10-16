import { AIMessage, type BaseMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { Db } from '@/core/storage/db';
import { readStoryMeta, type StoryMeta } from '@/core/story/story-meta';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { ChatStore, GM_CHAT } from '../chat-store';
import { createGmAgent, type GmAgentOptions } from '../gm-agent';
import { estimateTokens, inputMessage, sourceOf } from '../messages';
import { NothingToRetryError, retryTurn, runTurn } from '../run-turn';
import { TurnInProgressError } from '../turn-lock';
import { lastToolText, systemText } from '../testing/demo-scenario';
import { ScriptedChatModel, type Step } from '../testing/scripted-model';
import { TranscriptStore } from '../transcript';
import { repairDanglingToolCalls } from './bug-guard';
import { compactHistory, SUMMARY_PREFIX, TRIMMED_NOTE } from './context';
import { TOOL_LIMIT_TEXT } from './tool-limit';

const TAVERN = 'loc:rusty-gear-tavern';
const CHARACTER = { id: 'c1', name: 'Aria', sex: null, age: null, race: 'human', description: 'A cartographer.' };

let env: TestEnv;
let db: Db;
let world: World;
let meta: StoryMeta;
let chats: ChatStore;
let transcript: TranscriptStore;
let model: ScriptedChatModel;

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
  model = new ScriptedChatModel([]);
});

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

function play(input: string, steps: Step[], options: Partial<GmAgentOptions> = {}, signal?: AbortSignal) {
  model.push(...steps);
  const agent = createGmAgent({ model, world, meta, ...options });
  return runTurn({ agent, world, chats, transcript, input, signal });
}

const types = (messages: BaseMessage[]) => messages.map((m) => m.getType());
const ai = (content: string, calls: { id: string; name: string; args?: Record<string, unknown> }[] = []) =>
  new AIMessage({
    content,
    tool_calls: calls.map((c) => ({ id: c.id, name: c.name, args: c.args ?? {}, type: 'tool_call' as const })),
  });
const tool = (id: string, text = 'ok') => new ToolMessage({ tool_call_id: id, content: text, name: 'move' });
const say = (text: string) => inputMessage(text, 'player');

describe('BugGuard', () => {
  it('drops unanswered tool calls and keeps the answered ones', () => {
    const messages = [
      say('a'),
      ai('', [{ id: 'c1', name: 'move' }]),
      say('b'),
      ai('Partly.', [
        { id: 'c2', name: 'take' },
        { id: 'c3', name: 'drop' },
      ]),
      tool('c2'),
      ai('', [{ id: 'c4', name: 'move' }]),
    ].map((m, i) => Object.assign(m, { id: `m${i}` }));
    const fixes = repairDanglingToolCalls(messages);
    expect(fixes.map((f) => [f.getType(), f.id])).toEqual([
      ['remove', 'm1'],
      ['ai', 'm3'],
      ['remove', 'm5'],
    ]);
    expect((fixes[1] as AIMessage).tool_calls?.map((c) => c.id)).toEqual(['c2']);
    expect(repairDanglingToolCalls([say('a'), ai('', [{ id: 'c', name: 'move' }]), tool('c')])).toEqual([]);
  });

  it('repairs a turn that was cut off in the middle of a tool call', async () => {
    const abort = new AbortController();
    const embed = async () => {
      abort.abort(new Error('app killed'));
      await new Promise((resolve) => setTimeout(resolve, 20));
      return new Float32Array(1536);
    };
    await expect(
      play('Who is Mags?', [{ toolCalls: [{ name: 'search_lore', args: { query: 'Mags' } }] }], { embed }, abort.signal),
    ).rejects.toThrow();
    // saved step by step: the input and the call are there, the result never came
    const cut = await chats.load(GM_CHAT);
    expect(types(cut)).toEqual(['human', 'ai']);
    expect((cut[1] as AIMessage).tool_calls).toHaveLength(1);

    const result = await play('Who is Mags?', [
      (messages) => {
        // the model never sees the dangling call
        expect(types(messages.slice(1))).toEqual(['human', 'human']);
        return { text: 'Mags runs the crew.' };
      },
    ]);
    expect(result.reply).toBe('Mags runs the crew.');
    expect(types(await chats.load(GM_CHAT))).toEqual(['human', 'human', 'ai']);
  });
});

describe('Guard', () => {
  const verdict = (v: 'pass' | 'reject', reply = '') => ({ text: JSON.stringify({ verdict: v, reason: 'r', reply }) });

  it('rejects a declared fact with its own reply; the master is not called', async () => {
    const guardModel = new ScriptedChatModel([
      verdict('reject', 'You blink. The only sword here is the one in your imagination.'),
    ]);
    const result = await play('I draw my legendary sword.', [], { guardModel });
    expect(model.calls).toHaveLength(0);
    expect(result).toMatchObject({
      rejected: true,
      reply: 'You blink. The only sword here is the one in your imagination.',
      events: [],
    });
    // the guard saw the world rules, the state and the input
    const [system, user] = guardModel.calls[0];
    expect(system.text).toMatch(/# World rules\n- Era: industrial gaslight age/);
    expect(user.text).toMatch(/# Current state\nPlayer: Aria/);
    expect(user.text).toMatch(/# Player input\nI draw my legendary sword\.$/);
    // the master's chat keeps the exchange; the player saw both lines
    expect(types(await chats.load(GM_CHAT))).toEqual(['human', 'ai']);
    expect((await transcript.lines()).map((l) => [l.speaker, l.extra])).toEqual([
      ['player', {}],
      ['gm', { guard: 'reject' }],
    ]);
  });

  it('passes an attempt to the master, with the last GM words as context', async () => {
    const guardModel = new ScriptedChatModel([verdict('pass'), verdict('pass')]);
    await play('Hello.', [{ text: 'Mags nods.' }], { guardModel });
    const result = await play('I ask Mags about the plan.', [{ text: 'She taps the map.' }], { guardModel });
    expect(result).toMatchObject({ rejected: false, reply: 'She taps the map.' });
    expect(guardModel.calls[1][1].text).toContain("# The Game Master's last words\nMags nods.");
  });

  it('does not screen system commands', async () => {
    const guardModel = new ScriptedChatModel([]);
    model.push({ text: 'The story begins.' });
    const agent = createGmAgent({ model, world, meta, guardModel });
    const result = await runTurn({ agent, world, chats, transcript, input: 'session_start', source: 'system' });
    expect(guardModel.calls).toHaveLength(0);
    expect(result.reply).toBe('The story begins.');
    expect(sourceOf((await chats.load(GM_CHAT))[0])).toBe('system');
    expect((await transcript.lines()).map((l) => l.speaker)).toEqual(['gm']); // the command is not shown
  });

  it('lets the input through when the check fails', async () => {
    const onGuardError = vi.fn();
    const broken = new ScriptedChatModel([{ text: 'not json' }]);
    const result = await play('I look around.', [{ text: 'Smoke.' }], { guardModel: broken, onGuardError });
    expect(result.reply).toBe('Smoke.');
    expect(onGuardError).toHaveBeenCalledOnce();
    const silent = new ScriptedChatModel([]); // throws: no step
    expect((await play('Again.', [{ text: 'Still smoke.' }], { guardModel: silent, onGuardError })).reply).toBe(
      'Still smoke.',
    );
  });
});

/** A long history: `turns` turns of player / tool call / result / reply. */
function longHistory(turns: number, size = 2_000): BaseMessage[] {
  const filler = 'x'.repeat(size);
  return Array.from({ length: turns }, (_, t) => [
    say(`Turn ${t}: ${filler}`),
    ai('', [{ id: `c${t}`, name: 'inspect', args: { target: 'crate' } }]),
    tool(`c${t}`, filler),
    ai(`Reply ${t}. ${filler}`),
  ]).flat();
}

describe('Context', () => {
  it('summarizes the older part at the threshold, never splitting a call from its result', async () => {
    const summarizer = new ScriptedChatModel([{ text: 'Aria met Mags and studied the crate.' }]);
    const history = [...longHistory(30), say('Now?')];
    expect(estimateTokens(history)).toBeGreaterThan(30_000);
    const next = (await compactHistory(history, { model: summarizer }))!;
    expect(next[0].text).toBe(`${SUMMARY_PREFIX}\nAria met Mags and studied the crate.`);
    expect(sourceOf(next[0])).toBe('summary');
    expect(ToolMessage.isInstance(next[1])).toBe(false);
    expect(next.at(-1)!.text).toBe('Now?');
    expect(estimateTokens(next)).toBeLessThan(10_000);
    // the summarizer got a readable transcript of what was cut
    const transcriptText = summarizer.calls[0][1].text;
    expect(transcriptText).toMatch(/^Player: Turn 0: x+\n\[GM action\] inspect \{"target":"crate"\}\n\[Result\] x+\nGM: Reply 0/);
  });

  it('leaves a short history alone', async () => {
    expect(await compactHistory(longHistory(2), { model: new ScriptedChatModel([]) })).toBeUndefined();
  });

  it('drops the older part with a note when summarizing fails after retries', async () => {
    const onError = vi.fn();
    const failing = new ScriptedChatModel([{ text: '' }]); // empty, then no step: throws
    const next = (await compactHistory([...longHistory(30), say('Now?')], { model: failing, onError }))!;
    expect(next[0].text).toBe(TRIMMED_NOTE);
    expect(onError).toHaveBeenCalledTimes(2);
    expect(estimateTokens(next)).toBeLessThan(10_000);
  });

  it('keeps the current turn whole even when it is long', async () => {
    const big = 'z'.repeat(12_000);
    const current = [
      say('Big turn'),
      ...[1, 2, 3].flatMap((n) => [ai('', [{ id: `b${n}`, name: 'inspect' }]), tool(`b${n}`, big)]),
    ]; // one turn, ~9K tokens: more than the 8K tail kept after a summary
    const next = (await compactHistory([...longHistory(30), ...current], { model: new ScriptedChatModel([{ text: 'S' }]) }))!;
    expect(next.slice(1).map((m) => m.text)).toEqual(current.map((m) => m.text));
  });

  it('never goes over the hard ceiling, cutting old messages and then long tool results', async () => {
    const huge = [say('Look.'), ai('', [{ id: 'c', name: 'inspect' }]), tool('c', 'y'.repeat(60_000))];
    const next = (await compactHistory([...longHistory(5), ...huge], { triggerTokens: 1e9, maxTokens: 5_000 }))!;
    expect(next[0].text).toBe(TRIMMED_NOTE);
    expect(types(next.slice(1))).toEqual(['human', 'ai', 'tool']);
    expect(next.at(-1)!.text).toMatch(/^y{4000}… \[cut\]$/);
    expect(estimateTokens(next)).toBeLessThanOrEqual(5_000);
  });

  it('rewrites the stored history in a turn; the transcript keeps everything', async () => {
    await chats.append(GM_CHAT, longHistory(30));
    const summaryModel = new ScriptedChatModel([{ text: 'Long ago, things happened.' }]);
    const result = await play('What now?', [
      (messages) => {
        expect(messages[1].text).toBe(`${SUMMARY_PREFIX}\nLong ago, things happened.`);
        return { text: 'Now, this.' };
      },
    ], { summaryModel });
    expect(result.reply).toBe('Now, this.');
    const stored = await chats.load(GM_CHAT);
    expect(stored[0].text).toBe(`${SUMMARY_PREFIX}\nLong ago, things happened.`);
    expect(stored.length).toBeLessThan(longHistory(30).length / 3);
    expect(stored.at(-1)!.text).toBe('Now, this.');
    const rows = await db.all<{ seq: number }>('SELECT seq FROM model_messages ORDER BY seq');
    expect(rows.map((r) => Number(r.seq))).toEqual(stored.map((_, i) => i + 1));
    expect((await transcript.lines()).map((l) => l.text)).toEqual(['What now?', 'Now, this.']);
  });
});

describe('ToolLimit', () => {
  const inspect = { toolCalls: [{ name: 'inspect', args: { target: 'crate' } }] };

  it('allows 8 tool calls per turn, then asks for words', async () => {
    const result = await play('I search everything.', [
      ...Array.from({ length: 9 }, () => inspect),
      (messages) => ({ text: lastToolText(messages) === TOOL_LIMIT_TEXT ? 'You stop searching.' : '?' }),
    ]);
    expect(result).toMatchObject({ reply: 'You stop searching.', cutShort: false });
    const results = result.messages.filter((m) => ToolMessage.isInstance(m));
    expect(results).toHaveLength(9);
    expect(results.filter((m) => m.text === TOOL_LIMIT_TEXT)).toHaveLength(1);
  });

  it('cuts one reply with too many calls and runs the first ones', async () => {
    const result = await play('I grab it all.', [
      { toolCalls: Array.from({ length: 10 }, () => ({ name: 'inspect', args: { target: 'crate' } })) },
      { text: 'Enough.' },
    ]);
    const texts = result.messages.filter((m) => ToolMessage.isInstance(m)).map((m) => m.text);
    expect(texts.filter((t) => t === TOOL_LIMIT_TEXT)).toHaveLength(2);
    expect(texts.filter((t) => t.startsWith('crate'))).toHaveLength(8);
    expect(result.reply).toBe('Enough.');
  });

  it('ends the turn if the model keeps calling tools after the limit', async () => {
    const result = await play('Loop.', [
      { text: 'Searching…', toolCalls: Array.from({ length: 8 }, () => ({ name: 'inspect', args: { target: 'crate' } })) },
      inspect,
      inspect,
    ]);
    expect(result).toMatchObject({ cutShort: true, reply: 'Searching…' });
    expect(model.remaining).toBe(0);
    // every call in the stored history has its result
    const stored = await chats.load(GM_CHAT);
    const calls = stored.flatMap((m) => (AIMessage.isInstance(m) ? (m.tool_calls ?? []).map((c) => c.id) : []));
    const answered = new Set(stored.flatMap((m) => (ToolMessage.isInstance(m) ? [m.tool_call_id] : [])));
    expect(calls.every((id) => answered.has(id!))).toBe(true);
  });

  it('also limits a reply whose other calls had broken arguments', async () => {
    const result = await play('I grab it all.', [
      {
        toolCalls: Array.from({ length: 10 }, () => ({ name: 'inspect', args: { target: 'crate' } })),
        invalidToolCalls: [{ name: 'set_relation', args: '{x' }],
      },
      { text: 'Enough.' },
    ]);
    const texts = result.messages.filter((m) => ToolMessage.isInstance(m)).map((m) => m.text);
    expect(texts.filter((t) => t.startsWith('crate'))).toHaveLength(8);
    expect(texts.filter((t) => t === TOOL_LIMIT_TEXT)).toHaveLength(2);
    expect(texts.filter((t) => t.startsWith('Error: the arguments of set_relation'))).toHaveLength(1);
    expect(result.reply).toBe('Enough.');
  });

  it('counts only this turn', async () => {
    await play('First.', [...Array.from({ length: 6 }, () => inspect), { text: 'One.' }]);
    const result = await play('Second.', [...Array.from({ length: 6 }, () => inspect), { text: 'Two.' }]);
    expect(result.messages.filter((m) => m.text === TOOL_LIMIT_TEXT)).toHaveLength(0);
    expect(systemText(model.calls.at(-1)!)).toContain('# World state');
  });
});

describe('history', () => {
  it('saves the input before the model answers', async () => {
    const slow = new ScriptedChatModel([]);
    await expect(
      runTurn({ agent: createGmAgent({ model: slow, world, meta }), world, chats, transcript, input: 'Hi.' }),
    ).rejects.toThrow(/no step 1/);
    expect((await chats.load(GM_CHAT)).map((m) => m.text)).toEqual(['Hi.']);
    expect(HumanMessage.isInstance((await chats.load(GM_CHAT))[0])).toBe(true);
  });

  it('keeps every piece of the master\'s text, in order, in the reply and the transcript', async () => {
    const result = await play('I take the lantern.', [
      { text: 'You reach for the lantern.', toolCalls: [{ name: 'set_relation', args: { a: 'player', relation: 'has_item', b: 'brass lantern' } }] },
      { text: 'It is warm.' },
    ]);
    expect(result.reply).toBe('You reach for the lantern.\n\nIt is warm.');
    expect((await transcript.lines()).map((l) => [l.speaker, l.text])).toEqual([
      ['player', 'I take the lantern.'],
      ['gm', 'You reach for the lantern.'],
      ['gm', 'It is warm.'],
    ]);
  });

  it('runs one turn at a time per playthrough', async () => {
    model.push({ text: 'One.' });
    const agent = createGmAgent({ model, world, meta });
    const first = runTurn({ agent, world, chats, transcript, input: 'a' });
    await expect(runTurn({ agent, world, chats, transcript, input: 'b' })).rejects.toThrow(TurnInProgressError);
    expect((await first).reply).toBe('One.');
    // the lock is released after a turn, failed or not
    await expect(runTurn({ agent, world, chats, transcript, input: 'c' })).rejects.toThrow(/no step/);
    model.push({ text: 'Two.' });
    expect((await retryTurn({ agent, world, chats, transcript })).reply).toBe('Two.');
  });
});

describe('retryTurn', () => {
  it('finishes a failed turn without sending the input again, under the same turn number', async () => {
    const agent = createGmAgent({ model, world, meta });
    await expect(runTurn({ agent, world, chats, transcript, input: 'Hello?' })).rejects.toThrow(/no step/);
    model.push({ toolCalls: [{ name: 'set_relation', args: { a: 'player', relation: 'located_in', b: 'Lamplight Row' } }] }, { text: 'Yes?' });
    const result = await retryTurn({ agent, world, chats, transcript });
    expect(result).toMatchObject({ turn: 1, reply: 'Yes?' });
    expect(result.events.map((e) => [e.kind, e.turn])).toEqual([['player.moved', 1]]);
    expect((await chats.load(GM_CHAT)).map((m) => `${m.getType()}:${m.text}`)).toEqual([
      'human:Hello?',
      'ai:',
      'tool:The player is now in Lamplight Row [loc:lamplight-row]. The world state shows the new scene.',
      'ai:Yes?',
    ]);
    expect((await transcript.lines()).map((l) => l.text)).toEqual(['Hello?', 'Yes?']);
  });

  it('continues a turn cut off after a tool ran; its events stay in one turn', async () => {
    const abort = new AbortController();
    const embed = async () => {
      abort.abort(new Error('app killed'));
      await new Promise((resolve) => setTimeout(resolve, 20));
      return new Float32Array(1536);
    };
    await expect(
      play(
        'Look around and ask about Mags.',
        [
          { text: 'You glance around.', toolCalls: [{ name: 'note', args: { text: 'Asked about Mags.' } }] },
          { toolCalls: [{ name: 'search_lore', args: { query: 'Mags' } }] },
        ],
        { embed },
        abort.signal,
      ),
    ).rejects.toThrow();
    const agent = createGmAgent({ model, world, meta });
    model.push((messages) => {
      // BugGuard dropped the unanswered search; the note and its result are still there
      expect(messages.slice(1).map((m) => m.getType())).toEqual(['human', 'ai', 'tool']);
      return { text: 'Mags runs the crew.' };
    });
    const result = await retryTurn({ agent, world, chats, transcript });
    expect(result.turn).toBe(1);
    expect(result.reply).toBe('You glance around.\n\nMags runs the crew.');
    expect(result.events.map((e) => e.kind)).toEqual(['note.added']);
    expect((await transcript.lines()).map((l) => l.text)).toEqual([
      'Look around and ask about Mags.',
      'You glance around.',
      'Mags runs the crew.',
    ]);
  });

  it('refuses when the last turn already has its answer', async () => {
    await play('Hi.', [{ text: 'Hello.' }]);
    await expect(retryTurn({ agent: createGmAgent({ model, world, meta }), world, chats, transcript })).rejects.toThrow(
      NothingToRetryError,
    );
  });
});
