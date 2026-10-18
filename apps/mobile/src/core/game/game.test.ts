import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, type BaseMessage, HumanMessage } from '@langchain/core/messages';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ChatStore } from '@/core/agent/chat-store';
import { ScriptedChatModel } from '@/core/agent/testing/scripted-model';
import type { CloudSocket } from '@/core/net/cloud-socket';
import type { ServerEvent } from '@/core/protocol/messages';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { Game, type GameServices, isStoryChannel } from './game';

const ELF = { id: 'c1', name: 'Elaria', sex: 'female', age: 212, race: 'elf', description: 'An elf archer.' };
const ADAPTED = { name: 'Elaria', race: 'human', age: 28, description: 'An archer from a far island.' };

let env: TestEnv;
let game: Game;
let master: ScriptedChatModel;
let character: ScriptedChatModel;
let npc: ScriptedChatModel;
let listeners: Set<(event: ServerEvent) => void>;
/** The background generator's model for the next `open()`; none by default. */
let generator: BaseChatModel | undefined;
/** The entity memory's model for the next `open()`; none by default. */
let memory: BaseChatModel | undefined;

async function open(): Promise<Game> {
  const socket = {
    onEvent(listener: (event: ServerEvent) => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  } as unknown as CloudSocket;
  const services: GameServices = {
    library: env.library,
    socket,
    models: () => ({
      model: master,
      characterModel: character,
      actorModel: () => npc,
      generatorModel: generator,
      memoryModel: memory,
    }),
  };
  const saves = await env.library.repository.listSaves();
  return Game.open(services, saves[0].id);
}

beforeEach(async () => {
  env = await makeTestEnv();
  await env.library.importer.import(DEMO_STORY);
  await env.library.repository.addCharacter({ ...ELF, createdAt: '2026-10-15T00:00:00.000Z' });
  await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: 'c1', title: 'Heist' });
  master = new ScriptedChatModel([]);
  character = new ScriptedChatModel([]);
  npc = new ScriptedChatModel([]);
  listeners = new Set();
  generator = undefined;
  memory = undefined;
  game = await open();
});

afterEach(async () => {
  await game?.close();
  await env?.cleanup();
});

describe('Game', () => {
  it('goes from a new save to turns: start, choose, play', async () => {
    expect(await game.phase()).toBe('new');
    character.push({ text: JSON.stringify({ adapted: ADAPTED, alternatives: [], rationale: 'No elves here.' }) });
    const proposal = await game.start();
    expect(proposal.options[0]).toEqual(ADAPTED);
    expect(character.calls[0][1].text).toContain('Race: elf'); // the save's character from app.db
    expect(await game.phase()).toBe('adapting');
    expect(await game.proposal()).toEqual(proposal);

    master.push({ text: 'The tavern smokes.' });
    expect((await game.choose(0)).reply).toBe('The tavern smokes.');
    expect(await game.phase()).toBe('playing');

    master.push({ text: 'Mags nods.' });
    expect((await game.send('I nod to Mags.')).reply).toBe('Mags nods.');
    expect((await game.lines()).map((l) => l.speaker)).toEqual(['gm', 'player', 'gm', 'player', 'gm']);
    expect(await game.unfinished()).toBe(false);
    expect(await game.recover()).toBeUndefined();
  });

  it('recovers after a failure at any step', async () => {
    character.push({ text: 'oops' });
    await expect(game.start()).rejects.toThrow();
    character.push({ text: JSON.stringify({ adapted: ADAPTED, alternatives: [], rationale: 'Ok.' }) });
    expect(await game.recover()).toMatchObject({ round: 1 }); // the proposal
    await expect(game.choose(0)).rejects.toThrow(/no step/); // the master is offline
    expect([await game.phase(), await game.unfinished()]).toEqual(['playing', true]);
    master.push({ text: 'It begins.' });
    expect(await game.recover()).toMatchObject({ reply: 'It begins.' });
    await expect(game.send('Hm?')).rejects.toThrow(/no step/);
    expect(await game.unfinished()).toBe(true);
    master.push({ text: 'Yes.' });
    expect(await game.recover()).toMatchObject({ reply: 'Yes.' });
  });

  it('tells what is needed to go on, so a failed start or opening never leaves it stuck', async () => {
    expect(await game.nextStep()).toBe('start');
    character.push({ text: 'garbage' });
    await expect(game.start()).rejects.toThrow();
    await game.close();
    game = await open(); // the app restarted after the failed proposal
    expect([await game.phase(), await game.proposal(), await game.nextStep()]).toEqual(['adapting', undefined, 'start']);
    character.push({ text: JSON.stringify({ adapted: ADAPTED, alternatives: [], rationale: 'Ok.' }) });
    await game.recover();
    expect(await game.nextStep()).toBeUndefined(); // waiting for the player's choice

    // chosen, but the app died before the master's first turn began
    await game.world.adaptCharacter({ adapted: ADAPTED, choice: 'adapted' });
    expect([await game.phase(), await game.nextStep()]).toEqual(['opening', 'open']);
    master.push({ text: 'It begins.' });
    expect(await game.recover()).toMatchObject({ reply: 'It begins.' });
    expect(await game.nextStep()).toBeUndefined();

    await expect(game.send('Hm?')).rejects.toThrow(/no step/);
    expect(await game.nextStep()).toBe('retry');
  });

  it('survives reopening: the same phase and lines', async () => {
    character.push({ text: JSON.stringify({ adapted: ADAPTED, alternatives: [], rationale: 'Ok.' }) });
    await game.start();
    await game.close();
    game = await open();
    expect(await game.phase()).toBe('adapting');
    expect((await game.lines()).map((l) => l.extra.kind)).toEqual(['character_proposal']);
  });

  it('knows what each action cost, from the done events of its own generations', async () => {
    character.push({ text: JSON.stringify({ adapted: ADAPTED, alternatives: [], rationale: 'Ok.' }) });
    await game.start();
    master.push({ text: 'It begins.' });
    await game.choose(0);
    const emit = (event: ServerEvent) => listeners.forEach((l) => l(event));
    const routed = (requestId: string, sessionId = game.save.id) => ({ v: 1 as const, requestId, sessionId, channel: 'gm' });
    const done = (requestId: string, costUsd: number, sessionId?: string): ServerEvent => ({
      ...routed(requestId, sessionId),
      type: 'done',
      message: { role: 'assistant', content: 'x' },
      finishReason: 'stop',
      usage: { input: 9000, output: 300, cacheRead: 3000, costUsd },
    });
    master.push(() => {
      emit(done('old', 0.5)); // repeated after resume: not this action's
      emit({ ...routed('r1'), type: 'status', status: 'thinking' });
      emit({ ...routed('x', 'other-save'), type: 'status', status: 'thinking' });
      emit(done('x', 0.7, 'other-save')); // another playthrough
      emit(done('r1', 0.0025));
      return { text: 'Mags nods.' };
    });
    await game.send('I nod.');
    expect(game.lastUsage).toMatchObject({ calls: 1, input: 9000, cacheRead: 3000, costUsd: 0.0025 });
    expect(game.sessionUsage.calls).toBe(1);
  });

  it('streams only its own session, and names speakers', async () => {
    character.push({ text: JSON.stringify({ adapted: ADAPTED, alternatives: [], rationale: 'Ok.' }) });
    await game.start();
    const seen: ServerEvent[] = [];
    const off = game.onStream((event) => seen.push(event));
    const base = { v: 1 as const, requestId: 'r', channel: 'gm' };
    listeners.forEach((l) => l({ ...base, sessionId: game.save.id, type: 'delta', seq: 0, text: 'a' }));
    listeners.forEach((l) => l({ ...base, sessionId: 'other', type: 'delta', seq: 0, text: 'b' }));
    listeners.forEach((l) => l({ v: 1, type: 'heartbeat' }));
    off();
    expect(seen.map((e) => (e as { text: string }).text)).toEqual(['a']);
    expect(await game.speakerOf('npc:mags-holloway')).toBe('Mags Holloway');
    expect(await game.speakerOf('npc:item:crate')).toBe('crate');
    expect(await game.speakerOf('gm')).toBeUndefined();
    expect([isStoryChannel('gm'), isStoryChannel('npc:x'), isStoryChannel('guard'), isStoryChannel('system')]).toEqual([
      true,
      true,
      false,
      false,
    ]);
  });

  describe('the background generator (stage 6.2d)', () => {
    const MAGS = 'npc:mags-holloway';
    const EDIT = JSON.stringify({
      relations: [],
      cancelled: [],
      moves: [],
      edits: [{ target: MAGS, description: 'She no longer counts on the stranger.' }],
      additions: [],
    });
    const emit = (event: ServerEvent) => listeners.forEach((l) => l(event));

    /** A generator model that answers only when `release()` is called, and can be aborted. */
    function gatedGenerator() {
      let release!: () => void;
      const gate = new Promise<void>((resolve) => (release = resolve));
      const calls: BaseMessage[][] = [];
      const model = {
        invoke: async (messages: BaseMessage[], options: { signal: AbortSignal }) => {
          if (options.signal.aborted) throw new Error('aborted');
          calls.push(messages);
          const requestId = `gen-${calls.length}`;
          const routed = { v: 1 as const, requestId, sessionId: game.save.id, channel: 'generator' };
          emit({ ...routed, type: 'status', status: 'thinking' });
          await Promise.race([
            gate,
            new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(new Error('aborted')))),
          ]);
          emit({
            ...routed,
            type: 'done',
            message: { role: 'assistant', content: EDIT },
            finishReason: 'stop',
            usage: { input: 2000, output: 100, cacheRead: 0, costUsd: 0.001 },
          });
          return new AIMessage(EDIT);
        },
      } as unknown as BaseChatModel;
      return { model, calls, release };
    }

    async function playing(model: BaseChatModel) {
      generator = model;
      await game.close();
      game = await open();
      character.push({ text: JSON.stringify({ adapted: ADAPTED, alternatives: [], rationale: 'Ok.' }) });
      await game.start();
      master.push({ text: 'It begins.' });
      await game.choose(0);
    }

    it('follows a player turn without holding it up; the next turn waits for it', async () => {
      const gen = gatedGenerator();
      await playing(gen.model);
      expect(gen.calls).toHaveLength(0); // the opening is not the player's turn

      master.push({ text: 'Mags shrugs.' });
      expect((await game.send("I won't join.")).reply).toBe('Mags shrugs.');
      await vi.waitFor(() => expect(gen.calls).toHaveLength(1));
      expect((await game.world.state.entity(MAGS))?.description).not.toContain('no longer counts');

      master.push({ text: 'She looks away.' });
      const next = game.send('I wave.');
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(master.calls).toHaveLength(2); // still waiting for the generator
      gen.release();
      expect((await next).reply).toBe('She looks away.');

      expect(game.lastGeneration).toMatchObject({ rejected: [] });
      expect((await game.world.state.entity(MAGS))?.description).toBe('She no longer counts on the stranger.');
      // the master's next call already sees what the generator recorded
      expect(master.calls[2].at(-1)!.text).toContain('Mags Holloway changed: She no longer counts');
    });

    it("adds its cost to the turn's", async () => {
      const gen = gatedGenerator();
      await playing(gen.model);
      let notified = 0;
      game.onGenerated(() => (notified += 1));
      master.push({ text: 'Mags shrugs.' });
      await game.send("I won't join.");
      gen.release();
      await game.whenGenerated();

      expect(notified).toBe(1);
      expect(game.lastUsage?.byChannel.generator).toEqual({ calls: 1, costUsd: 0.001 });
      expect(game.sessionUsage.byChannel.generator).toEqual({ calls: 1, costUsd: 0.001 });
    });

    it('is stopped when the game closes', async () => {
      const gen = gatedGenerator();
      await playing(gen.model);
      master.push({ text: 'Mags shrugs.' });
      await game.send("I won't join.");
      await vi.waitFor(() => expect(gen.calls).toHaveLength(1));
      await game.close(); // must not wait for an answer that never comes
      expect(game.lastGeneration).toEqual({ error: 'aborted' });
      game = await open();
    });

    it('a turn whose background work was lost is finished when the game opens again, once (ADR 48)', async () => {
      const gen = gatedGenerator();
      await playing(gen.model);
      master.push({ text: 'Mags shrugs.' });
      await game.send("I won't join.");
      await vi.waitFor(() => expect(gen.calls).toHaveLength(1));
      await game.close(); // the app is killed: the generator never finished

      const again = new ScriptedChatModel([{ text: EDIT }]);
      generator = again;
      game = await open();
      await game.whenGenerated();
      expect(again.calls).toHaveLength(1);
      expect(again.calls[0][1].text).toContain("Player: I won't join.");
      expect((await game.world.state.entity(MAGS))?.description).toBe('She no longer counts on the stranger.');

      await game.close();
      game = await open(); // done now: not run a second time
      await game.whenGenerated();
      expect(again.calls).toHaveLength(1);
    });
  });

  it('folds an overflowing entity chat into its notes after the turn, in the background (ADR 45)', async () => {
    character.push({ text: JSON.stringify({ adapted: ADAPTED, alternatives: [], rationale: '.' }) });
    await game.start();
    master.push({ text: 'It begins.' });
    await game.choose(0);
    const MAGS = 'npc:mags-holloway';
    await game.close();
    // Mags has talked a lot before (her chat is past its window)
    const db = await env.library.saves.open(game.save.id);
    const filler = 'w'.repeat(1_000);
    await new ChatStore(db).append(
      MAGS,
      Array.from({ length: 12 }, (_, i) => [new HumanMessage(`Q${i} ${filler}`), new AIMessage(`A${i} ${filler}`)]).flat(),
    );
    await db.close();
    const notes = new ScriptedChatModel([{ text: JSON.stringify({ notes: [{ entity: MAGS, text: 'I sized up the elf.' }] }) }]);
    memory = notes;
    game = await open();

    npc.push({ text: '"Hm."' });
    master.push(
      { toolCalls: [{ name: 'react_entity', args: { target: MAGS, situation: 'The elf nods.' } }] },
      { text: 'She nods back.' },
    );
    expect((await game.send('I nod to Mags.')).reply).toBe('She nods back.');
    await game.whenGenerated();
    expect(notes.calls).toHaveLength(1);
    expect((await game.world.log.actorNotes(MAGS, 5)).map((n) => n.text)).toEqual(['I sized up the elf.']);
    const folded = notes.calls[0][1].text;
    expect(folded).toContain('- Happened: Q0 ');
    expect(folded).not.toContain('The elf nods.'); // the latest exchange stays in her chat
  });
});
