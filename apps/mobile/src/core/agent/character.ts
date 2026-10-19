/**
 * Character adaptation before the story begins (agent.md#старт-сессии, ADR 37): a chat of its
 * own with its own prompt — not the master's — that fits the player's character sheet to the
 * world rules and offers it as `propose_character`: the adapted character, up to two
 * alternatives and a short rationale. The player picks a card or asks for changes (up to three
 * proposals in all). The chat lives in `model_messages` ('character') only until the choice.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, HumanMessage, type SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import type { StoryMeta } from '../story/story-meta';
import type { CharacterSheet } from '../world/world';
import type { ChatStore } from './chat-store';
import { promptMessage, promptPart } from './cloud-prompt';
import { invokeWithin, jsonSchemaFormat, parseJsonAnswer, strictObject } from './structured';

export const CHARACTER_CHAT = 'character';
/** Proposals per adaptation: the third one can only be chosen from (owner, stage 4.4). */
export const MAX_PROPOSALS = 3;
export const MAX_ALTERNATIVES = 2;


const OPTION_SCHEMA = strictObject({
  name: { type: 'string' },
  race: { type: 'string' },
  age: { type: ['integer', 'null'] },
  description: { type: 'string' },
});

/** OpenAI-style `response_format` for the character model (passed to the provider as is). */
export const CHARACTER_RESPONSE_FORMAT = jsonSchemaFormat(
  'propose_character',
  strictObject({
    adapted: OPTION_SCHEMA,
    alternatives: { type: 'array', items: OPTION_SCHEMA },
    rationale: { type: 'string' },
  }),
);

const Option = z.object({
  name: z.string().min(1),
  race: z.string().min(1),
  age: z.number().int().positive().nullable().default(null),
  description: z.string().min(1),
});
export const Proposal = z.object({
  adapted: Option,
  alternatives: z.array(Option).default([]),
  rationale: z.string().default(''),
});

export type CharacterOption = z.infer<typeof Option>;

export interface CharacterProposal {
  /** 1-based: which proposal of this adaptation it is. */
  round: number;
  /** The adapted character first, then the alternatives. */
  options: CharacterOption[];
  rationale: string;
  /** False on the last round: only a card can be chosen. */
  canClarify: boolean;
}

/** The model answered something that is not a proposal (retry the same step). */
export class CharacterProposalError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'CharacterProposalError';
  }
}

export interface CharacterDeps {
  /** QueueChatModel with purpose 'character' and CHARACTER_RESPONSE_FORMAT. */
  model: BaseChatModel;
  chats: ChatStore;
  meta: Pick<StoryMeta, 'worldSummary' | 'worldConstraints'>;
  timeoutMs?: number;
}

/** The cloud's template `character` (ADR 57) with the story's world. */
export function characterSystemMessage(meta: CharacterDeps['meta']): SystemMessage {
  return promptMessage(
    promptPart('character', {
      WORLD_SUMMARY: meta.worldSummary.trim() || 'Not described.',
      WORLD_CONSTRAINTS: meta.worldConstraints.trim() || 'None given.',
    }),
  );
}

/** The sheet as the player filled it in (the first message of the chat). */
export function sheetText(sheet: CharacterSheet): string {
  const line = (label: string, value: unknown) =>
    value === undefined || value === null || value === '' ? '' : `${label}: ${value}`;
  return [
    'My character:',
    line('Name', sheet.name),
    line('Sex', sheet.sex),
    line('Age', sheet.age),
    line('Race', sheet.race),
    line('Description', sheet.description),
  ]
    .filter(Boolean)
    .join('\n');
}

/**
 * One step of the adaptation: the player's message (the sheet, or their wishes) joins the chat,
 * the model proposes, and both are saved only when the answer is a valid proposal.
 */
export async function proposeCharacter(
  deps: CharacterDeps,
  playerMessage: string,
  signal?: AbortSignal,
): Promise<CharacterProposal> {
  const history = await deps.chats.load(CHARACTER_CHAT);
  const round = history.filter((m) => AIMessage.isInstance(m)).length + 1;
  if (round > MAX_PROPOSALS) throw new CharacterProposalError('No more changes: choose one of the proposed characters.');
  const ask = new HumanMessage(playerMessage);
  const system = characterSystemMessage(deps.meta);
  const answer = await invokeWithin(deps.model, [system, ...history, ask], { timeoutMs: deps.timeoutMs ?? 60_000, signal });
  const proposal = parseProposal(answer.text, round);
  await deps.chats.append(CHARACTER_CHAT, [ask, new AIMessage(answer.text)]);
  return proposal;
}

/** The latest proposal of an unfinished adaptation (after an app restart), if any. */
export async function lastProposal(chats: ChatStore): Promise<CharacterProposal | undefined> {
  const answers = (await chats.load(CHARACTER_CHAT)).filter((m) => AIMessage.isInstance(m));
  return answers.length ? parseProposal(answers.at(-1)!.text, answers.length) : undefined;
}

export function parseProposal(text: string, round: number): CharacterProposal {
  let data: z.infer<typeof Proposal>;
  try {
    data = parseJsonAnswer(Proposal, text);
  } catch (error) {
    throw new CharacterProposalError('The character proposal could not be read.', { cause: error });
  }
  return {
    round,
    options: [data.adapted, ...data.alternatives.slice(0, MAX_ALTERNATIVES)],
    rationale: data.rationale.trim(),
    canClarify: round < MAX_PROPOSALS,
  };
}
