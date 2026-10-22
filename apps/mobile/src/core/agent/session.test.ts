import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { Db } from '@/core/storage/db';
import { readStoryMeta, type StoryMeta } from '@/core/story/story-meta';
import { World } from '@/core/world/world';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';
import { renderedText } from '@/platform/node/prompts';

import { CHARACTER_CHAT, CharacterProposalError, MAX_PROPOSALS } from './character';
import { ChatStore, GM_CHAT } from './chat-store';
import { createGmAgent } from './gm-agent';
import { sourceOf } from './messages';
import { beginSession, chooseCharacter, clarifyCharacter, PROPOSAL_CARD, type SessionDeps, startStory } from './session';
import { systemText } from './testing/demo-scenario';
import { ScriptedChatModel } from './testing/scripted-model';
import { TranscriptStore } from './transcript';
import { TurnInProgressError } from './turn-lock';

const TAVERN = 'loc:rusty-gear-tavern';
const ELF = {
  id: 'c1',
  name: 'Elaria Moonwhisper',
  sex: 'female',
  age: 212,
  race: 'elf',
  description: 'A high elf archer who speaks with trees and casts moonlight spells.',
};

const option = (name: string, race: string, description: string, age: number | null = 28) => ({
  name,
  race,
  age,
  description,
});
const ADAPTED = option(
  'Elaria Moon',
  'human',
  'A sharp-eyed archer from a distant island, pale-haired and tall, who reads people like weather. Carries a short bow and a tin of lamp oil.',
);
const ALT_1 = option('Elaria Moon', 'human', 'A stage illusionist who once played the elf queen in a touring show.');
const ALT_2 = option('Ellie Moonwhistle', 'human', 'A colonial botanist with a talent for poisons and long silences.');
const proposal = (adapted: object, alternatives: object[], rationale: string) => ({
  text: JSON.stringify({ adapted, alternatives, rationale }),
});
/** Two alternatives: a full proposal of three cards (stage T.2). */
const ALTS = [ALT_1, ALT_2];

let env: TestEnv;
let db: Db;
let world: World;
let meta: StoryMeta;
let deps: SessionDeps;
let characterModel: ScriptedChatModel;
let master: ScriptedChatModel;

beforeEach(async () => {
  env = await makeTestEnv();
  await env.library.importer.import(DEMO_STORY);
  await env.library.repository.addCharacter({ ...ELF, createdAt: '2026-10-15T00:00:00.000Z' });
  const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: 'c1', title: 't' });
  db = await env.library.saves.open(save.id);
  world = await World.open(db);
  meta = await readStoryMeta(db);
  characterModel = new ScriptedChatModel([]);
  master = new ScriptedChatModel([]);
  deps = { world, chats: new ChatStore(db), transcript: new TranscriptStore(db), meta, characterModel };
});

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

const agent = () => createGmAgent({ model: master, world, meta });

describe('session start: an elf in a world without elves', () => {
  it('adapts the character, offers alternatives, and opens the scene after the choice', async () => {
    const firstAnswer = proposal(
      ADAPTED,
      [ALT_1, ALT_2],
      'Vellmoor has no elves, but it has islands far stranger than its maps admit.',
    );
    characterModel.push(firstAnswer);
    const first = await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });

    expect(first).toEqual({
      round: 1,
      options: [ADAPTED, ALT_1, ALT_2],
      rationale: 'Vellmoor has no elves, but it has islands far stranger than its maps admit.',
      canClarify: true,
      maxRounds: 4,
    });
    // a chat of its own: the world rules and the player's sheet, not the master's prompt
    const [system, sheet] = characterModel.calls[0];
    expect(renderedText(system)).toMatch(/^You prepare the player's character/);
    expect(renderedText(system)).toContain('Do not add non-human peoples');
    expect(renderedText(system)).toContain('# The world\nThe Brass Vault is set in Vellmoor');
    expect(sheet.text).toBe(
      'My character:\nName: Elaria Moonwhisper\nSex: female\nAge: 212\nRace: elf\n' +
        'Description: A high elf archer who speaks with trees and casts moonlight spells.',
    );
    expect(master.calls).toHaveLength(0);
    expect((await world.scene()).location?.id).toBe(TAVERN);

    // the player asks for magic; the next proposal keeps the world's rules
    characterModel.push(proposal(ALT_1, [ADAPTED, ALT_2], 'No magic here, but a fine illusion goes a long way.'));
    const second = await clarifyCharacter(deps, 'Can she keep some magic?');
    expect(second).toMatchObject({ round: 2, options: [ALT_1, ADAPTED, ALT_2], canClarify: true });
    expect(characterModel.calls[1].slice(1).map((m) => m.text)).toEqual([
      sheet.text,
      firstAnswer.text,
      'Can she keep some magic?',
    ]);

    // the choice: character.adapted, the adaptation chat is gone, the master opens the story
    master.push(
      {
        toolCalls: [
          { name: 'set_entity', args: { name: 'short bow', type: 'item', description: 'Yew, well oiled.' } },
          { name: 'set_relation', args: { a: 'player', relation: 'has_item', b: 'short bow' } },
        ],
      },
      (messages) => {
        const state = systemText(messages);
        return {
          text: state.includes('Player: Elaria Moon (human, female, age 28)')
            ? 'Smoke hangs low in the Rusty Gear. Mags Holloway looks up from the plan: "You must be the archer."'
            : '?',
        };
      },
    );
    const scene = await chooseCharacter({ ...deps, agent: agent() }, 1);

    expect(scene.reply).toBe(
      'Smoke hangs low in the Rusty Gear. Mags Holloway looks up from the plan: "You must be the archer."',
    );
    const kinds = (await world.log.recent(10)).map((e) => e.kind);
    expect(kinds).toEqual(['session.started', 'character.adapted', 'entity.added', 'item.taken']);
    const adaptedEvent = (await world.log.recent(10))[1];
    expect(adaptedEvent.payload).toMatchObject({ adapted: ADAPTED, choice: 'alternative 1', original: ELF });
    const player = (await world.state.player())!;
    expect(player).toMatchObject({ name: 'Elaria Moon', description: ADAPTED.description });
    expect(player.props).toMatchObject({ race: 'human', age: 28, original: ELF });
    expect((await world.scene()).inventory.map((e) => e.name)).toEqual(['short bow']);

    expect(await deps.chats.load(CHARACTER_CHAT)).toEqual([]);
    const gm = await deps.chats.load(GM_CHAT);
    expect(sourceOf(gm[0])).toBe('system');
    expect(gm[0].text).toMatch(
      /^\[session_start\] "The Brass Vault" begins now\.\nThe player's character: Elaria Moon, human, 28 years old\./,
    );

    const lines = await deps.transcript.lines();
    expect(lines.map((l) => [l.speaker, l.extra.kind ?? null])).toEqual([
      ['gm', PROPOSAL_CARD],
      ['player', null],
      ['gm', PROPOSAL_CARD],
      ['player', 'character_choice'],
      ['gm', null],
    ]);
    expect(lines[0].extra).toMatchObject({ round: 1, options: [ADAPTED, ALT_1, ALT_2], canClarify: true, maxRounds: 4 });
    expect(lines[3]).toMatchObject({ text: 'Elaria Moon, human', extra: { choice: 'alternative 1' } });
  });
});

describe('session start rules', () => {
  it('allows three changes (four proposals); the last one can only be chosen from', async () => {
    expect(MAX_PROPOSALS).toBe(4);
    for (let i = 0; i < MAX_PROPOSALS; i++) characterModel.push(proposal(ADAPTED, ALTS, `Round ${i + 1}.`));
    await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    await clarifyCharacter(deps, 'Older, please.');
    await clarifyCharacter(deps, 'Younger after all.');
    const last = await clarifyCharacter(deps, 'Taller.');
    expect(last).toMatchObject({ round: 4, canClarify: false, maxRounds: 4, options: [ADAPTED, ALT_1, ALT_2] });
    await expect(clarifyCharacter(deps, 'One more?')).rejects.toThrow(CharacterProposalError);
    expect(characterModel.remaining).toBe(0);
  });

  it('resumes after a restart: the pending proposal comes back without a new call', async () => {
    characterModel.push(proposal(ADAPTED, ALTS, 'Welcome.'));
    const first = await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    const again = await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    expect(again).toEqual(first);
    expect(characterModel.calls).toHaveLength(1);
    expect(await world.log.count('session.started')).toBe(1);
  });

  it('keeps nothing from an unreadable proposal, so the step can be retried', async () => {
    characterModel.push({ text: 'Sure! Here is your elf…' }, proposal(ADAPTED, ALTS, 'Welcome.'));
    await expect(beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF })).rejects.toThrow(
      CharacterProposalError,
    );
    expect(await deps.chats.load(CHARACTER_CHAT)).toEqual([]);
    expect(await deps.transcript.lines()).toEqual([]);
    expect((await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF })).round).toBe(1);
  });

  it('keeps at most two alternatives and refuses a card that does not exist', async () => {
    characterModel.push(proposal(ADAPTED, [ALT_1, ALT_2, ALT_1], 'Many paths.'));
    const first = await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    expect(first.options).toHaveLength(3);
    await expect(chooseCharacter({ ...deps, agent: agent() }, 3)).rejects.toThrow(/no character #4/);
    expect(await world.log.count('character.adapted')).toBe(0);
  });

  it('asks once more when the model offered fewer than three characters, and keeps only the full answer', async () => {
    const short = proposal(ADAPTED, [], 'Welcome.');
    const full = proposal(ADAPTED, ALTS, 'Welcome, in three ways.');
    characterModel.push(short, full);
    const first = await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    expect(first.options).toEqual([ADAPTED, ALT_1, ALT_2]);
    expect(first.rationale).toBe('Welcome, in three ways.');
    // the retry sees the short answer and the reminder
    expect(characterModel.calls[1].slice(-2).map((m) => m.text)).toEqual([
      short.text,
      'You offered 1 character. Offer exactly three: the adapted character and two alternatives. Answer with the whole proposal again.',
    ]);
    // the chat keeps the sheet and the full answer only
    expect((await deps.chats.load(CHARACTER_CHAT)).map((m) => m.text).slice(1)).toEqual([full.text]);
    expect((await deps.transcript.lines()).map((l) => l.text)).toEqual(['Welcome, in three ways.']);
  });

  it('gives up after a second short answer: nothing is kept, the step can be retried', async () => {
    characterModel.push(proposal(ADAPTED, [ALT_1], 'Two.'), proposal(ADAPTED, [], 'One.'), proposal(ADAPTED, ALTS, 'Three.'));
    await expect(beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF })).rejects.toThrow(/three characters/);
    expect(await deps.chats.load(CHARACTER_CHAT)).toEqual([]);
    expect(await deps.transcript.lines()).toEqual([]);
    expect((await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF })).options).toHaveLength(3);
  });

  it('resumes a proposal saved before three cards were required, as it is', async () => {
    // a save from before stage T.2: one card in the chat
    characterModel.push(proposal(ADAPTED, ALTS, 'unused'));
    await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    await deps.chats.clear(CHARACTER_CHAT);
    await deps.chats.append(CHARACTER_CHAT, [new HumanMessage('sheet'), new AIMessage(proposal(ADAPTED, [], 'Old.').text)]);
    const again = await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    expect(again).toMatchObject({ round: 1, options: [ADAPTED], rationale: 'Old.', canClarify: true });
  });

  it('a card of an earlier proposal can still be chosen (stage T.3)', async () => {
    characterModel.push(proposal(ADAPTED, ALTS, 'First.'), proposal(ALT_1, [ADAPTED, ALT_2], 'Second.'));
    await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    await clarifyCharacter(deps, 'Different, please.');
    await expect(chooseCharacter({ ...deps, agent: agent() }, 0, 3)).rejects.toThrow(/no character #1/);
    master.push({ text: 'The story begins.' });
    await chooseCharacter({ ...deps, agent: agent() }, 2, 1);
    expect((await world.state.player())!.description).toBe(ALT_2.description);
    const choice = (await deps.transcript.lines()).find((l) => l.extra.kind === 'character_choice');
    expect(choice?.extra).toMatchObject({ choice: 'alternative 2', round: 1, index: 2 });
  });

  it('a double tap on a card adapts once and opens one scene', async () => {
    characterModel.push(proposal(ADAPTED, ALTS, 'Welcome.'));
    await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    master.push({ text: 'The story begins.' }, { text: 'Again?' });
    const both = await Promise.allSettled([
      chooseCharacter({ ...deps, agent: agent() }, 0),
      chooseCharacter({ ...deps, agent: agent() }, 1),
    ]);
    expect(both.map((r) => r.status)).toEqual(['fulfilled', 'rejected']);
    expect((both[1] as PromiseRejectedResult).reason).toBeInstanceOf(TurnInProgressError);
    expect(await world.log.count('character.adapted')).toBe(1);
    expect(master.calls).toHaveLength(1);
    // and later, too
    await expect(chooseCharacter({ ...deps, agent: agent() }, 0)).rejects.toThrow(CharacterProposalError);
  });

  it("shows the player's wishes once, even when the first try failed", async () => {
    characterModel.push(proposal(ADAPTED, ALTS, 'Welcome.'), { text: 'garbage' }, proposal(ALT_1, [ADAPTED, ALT_2], 'Taller now.'));
    await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    await expect(clarifyCharacter(deps, 'Taller, please.')).rejects.toThrow(CharacterProposalError);
    await clarifyCharacter(deps, 'Taller, please.');
    expect((await deps.transcript.lines()).map((l) => `${l.speaker}:${l.text}`)).toEqual([
      'gm:Welcome.',
      'player:Taller, please.',
      'gm:Taller now.',
    ]);
  });

  it('can retry the opening scene alone if it failed after the choice', async () => {
    characterModel.push(proposal(ADAPTED, ALTS, 'Welcome.'));
    await beginSession(deps, { storyId: DEMO_STORY_ID, character: ELF });
    await expect(chooseCharacter({ ...deps, agent: agent() }, 0)).rejects.toThrow(/no step 1/); // master offline
    expect(await world.log.count('character.adapted')).toBe(1);
    await expect(clarifyCharacter(deps, 'Wait!')).rejects.toThrow(/already chosen/);
    master.push({ text: 'The story begins.' });
    expect((await startStory({ ...deps, agent: agent() })).reply).toBe('The story begins.');
  });
});
