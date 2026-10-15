/**
 * The start of a playthrough (agent.md#старт-сессии): `session.started`, the character
 * adaptation in its own chat (character.ts), the player's choice → `character.adapted`, and the
 * master's first turn — the opening scene — on the system command `session_start`.
 * The UI calls these in order; each step can be retried after a failure.
 */
import type { StoryMeta } from '../story/story-meta';
import type { CharacterSheet, World } from '../world/world';
import {
  CHARACTER_CHAT,
  type CharacterDeps,
  type CharacterOption,
  type CharacterProposal,
  CharacterProposalError,
  lastProposal,
  proposeCharacter,
  sheetText,
} from './character';
import type { ChatStore } from './chat-store';
import type { GmAgent } from './gm-agent';
import { runTurn, type TurnResult } from './run-turn';
import type { TranscriptStore } from './transcript';

export interface SessionDeps {
  world: World;
  chats: ChatStore;
  transcript: TranscriptStore;
  meta: StoryMeta;
  /** QueueChatModel with purpose 'character' (queueModels().characterModel). */
  characterModel: CharacterDeps['model'];
  signal?: AbortSignal;
}

/** Transcript `extra.kind` of the proposal card and of the player's pick. */
export const PROPOSAL_CARD = 'character_proposal';
export const CHOICE_CARD = 'character_choice';

/**
 * Starts the playthrough and returns the first proposal. Idempotent: after a restart it returns
 * the pending proposal instead of starting again.
 */
export async function beginSession(
  deps: SessionDeps,
  input: { storyId: string; character: CharacterSheet },
): Promise<CharacterProposal> {
  const { world } = deps;
  if ((await world.log.count('session.started')) === 0) {
    const started = await world.startSession({
      storyId: input.storyId,
      character: input.character,
      startLocation: deps.meta.startLocation,
    });
    if (!started.ok) throw new Error(started.error);
  }
  if (await isAdapted(world)) throw new CharacterProposalError('The character is already chosen.');
  const pending = await lastProposal(deps.chats);
  if (pending) return pending;
  return propose(deps, sheetText(input.character));
}

/** The player's wishes for the character: a new proposal (not on the last round). */
export async function clarifyCharacter(deps: SessionDeps, wishes: string): Promise<CharacterProposal> {
  if (await isAdapted(deps.world)) throw new CharacterProposalError('The character is already chosen.');
  const pending = await lastProposal(deps.chats);
  if (!pending) throw new CharacterProposalError('There is no proposal to change yet.');
  if (!pending.canClarify) {
    throw new CharacterProposalError('No more changes: choose one of the proposed characters.');
  }
  await deps.transcript.append('player', wishes);
  return propose(deps, wishes);
}

/**
 * The player picked a card (0 = adapted, 1–2 = alternatives): `character.adapted`, the
 * adaptation chat is dropped, and the master opens the story.
 */
export async function chooseCharacter(
  deps: SessionDeps & { agent: GmAgent },
  index: number,
): Promise<TurnResult> {
  const pending = await lastProposal(deps.chats);
  const option = pending?.options[index];
  if (!pending || !option) throw new CharacterProposalError(`There is no character #${index + 1} to choose.`);
  const choice = index === 0 ? 'adapted' : `alternative ${index}`;
  const adapted = await deps.world.adaptCharacter({ adapted: option, choice });
  if (!adapted.ok) throw new Error(adapted.error);
  await deps.transcript.append('player', `${option.name}, ${option.race}`, { kind: CHOICE_CARD, choice });
  await deps.chats.clear(CHARACTER_CHAT);
  return startStory(deps);
}

/** The master's first turn: the opening scene. Retry it alone if it failed after the choice. */
export async function startStory(deps: SessionDeps & { agent: GmAgent }): Promise<TurnResult> {
  const player = await deps.world.state.player();
  if (!player) throw new Error('The session has not started.');
  return runTurn({
    agent: deps.agent,
    world: deps.world,
    chats: deps.chats,
    transcript: deps.transcript,
    input: sessionStartCommand(
      {
        name: player.name,
        race: String(player.props.race ?? ''),
        age: typeof player.props.age === 'number' ? player.props.age : null,
        description: player.description,
      },
      deps.meta,
    ),
    source: 'system',
    signal: deps.signal,
  });
}

/** The system command that opens the story (English, like all prompts). */
export function sessionStartCommand(character: CharacterOption, meta: Pick<StoryMeta, 'title'>): string {
  const race = [character.race, character.age ? `${character.age} years old` : ''].filter(Boolean).join(', ');
  return [
    `[session_start] "${meta.title}" begins now.`,
    `The player's character: ${character.name}${race ? `, ${race}` : ''}. ${character.description}`,
    'First give the player the few possessions their description mentions, with add_entity (place: inventory). ' +
      "Then open the story with a vivid scene at the player's location from the world state: where they are, " +
      'who is there and what is at stake, ending where the player can act.',
  ].join('\n');
}

async function propose(deps: SessionDeps, message: string): Promise<CharacterProposal> {
  const proposal = await proposeCharacter(
    { model: deps.characterModel, chats: deps.chats, meta: deps.meta },
    message,
    deps.signal,
  );
  await deps.transcript.append('gm', proposal.rationale, {
    kind: PROPOSAL_CARD,
    round: proposal.round,
    options: proposal.options,
    canClarify: proposal.canClarify,
  });
  return proposal;
}

async function isAdapted(world: World): Promise<boolean> {
  return (await world.log.count('character.adapted')) > 0;
}
