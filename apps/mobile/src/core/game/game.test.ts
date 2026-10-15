import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { ScriptedChatModel } from '@/core/agent/testing/scripted-model';
import type { CloudSocket } from '@/core/net/cloud-socket';
import type { ServerEvent } from '@/core/protocol/messages';
import type { Entity } from '@/core/world/model';
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
      npcModel: (_npc: Entity) => npc,
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
});
