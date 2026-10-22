import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { HumanMessage } from '@langchain/core/messages';
import { createMiddleware } from 'langchain';
import { z } from 'zod';

import type { World } from '../../world/world';
import { promptMessage, promptPart } from '../cloud-prompt';
import { LIKELIHOODS, type LuckOutcome, rollLuck, STAKES, SURE_LIKELIHOODS } from '../luck';
import { sourceOf } from '../messages';
import { renderWorldState } from '../prompt';
import { invokeWithin, jsonSchemaFormat, parseJsonAnswer, strictObject } from '../structured';
import { addDrafts, type ExpectedGap, type GuardHints, MAX_DRAFTS } from './guard';

const DRAFT_KINDS = ['npc', 'creature', 'other'] as const;
const CLAIM_KINDS = ['item', 'ability', 'fact'] as const;
/** How plausible a claim can be: never `certain` — then it would be in the world already. */
const CLAIM_LIKELIHOODS = LIKELIHOODS.filter((l) => l !== 'certain') as ['likely', 'possible', 'unlikely'];
/** Claims rolled for in one input, at most. */
export const MAX_CLAIMS = 2;
/** The story's luck when `meta.luck` is not given (as try_luck, ADR 47). */
const FAIR_LUCK = 0.5;

/** OpenAI-style `response_format` of the `pregen` call (passed to the provider as is). */
export const PREGEN_RESPONSE_FORMAT = jsonSchemaFormat(
  'pregen',
  strictObject({
    drafts: {
      type: 'array',
      items: strictObject({
        name: { type: 'string' },
        type: { type: 'string', enum: [...DRAFT_KINDS] },
        description: { type: 'string' },
        likelihood: { type: 'string', enum: [...LIKELIHOODS] },
      }),
    },
    claims: {
      type: 'array',
      items: strictObject({
        name: { type: 'string' },
        kind: { type: 'string', enum: [...CLAIM_KINDS] },
        likelihood: { type: 'string', enum: [...CLAIM_LIKELIHOODS] },
        stakes: { type: 'string', enum: [...STAKES] },
      }),
    },
  }),
);

export const Pregen = z.object({
  drafts: z
    .array(
      z.object({
        name: z.string().trim().min(1),
        type: z.enum(DRAFT_KINDS),
        description: z.string().default(''),
        likelihood: z.enum(LIKELIHOODS),
      }),
    )
    .default([]),
  claims: z
    .array(
      z.object({
        name: z.string().trim().min(1),
        kind: z.enum(CLAIM_KINDS),
        likelihood: z.enum(CLAIM_LIKELIHOODS),
        stakes: z.enum(STAKES),
      }),
    )
    .default([]),
});

export interface BeforeGeneratorOptions {
  /** QueueChatModel with purpose 'pregen' and PREGEN_RESPONSE_FORMAT. */
  model: BaseChatModel;
  world: World;
  /** The turn's hints: the Guard's decision in, the drafts' gaps and the checks out. */
  hints: GuardHints;
  /** The story's luck, 0..1 (`meta.luck`, ADR 47). */
  luck?: number;
  /** [0, 1); default Math.random. */
  random?: () => number;
  timeoutMs?: number;
  onError?: (error: unknown) => void;
}

/** What the master reads about a claim, by the luck's outcome (in the world state, this turn only). */
const CLAIM_TEXT: Record<LuckOutcome | 'refused', (name: string) => string> = {
  success: (name) =>
    `"${name}" is not in the world, but luck allows a modest find of this kind: narrate it only if it fits the place and the character, otherwise it is not there.`,
  failure: (name) => `"${name}" is not there. Narrate the attempt and that it finds nothing.`,
  mishap: (name) => `"${name}" is not there, and the attempt goes wrong. Narrate the mishap, in proportion.`,
  refused: (name) => `"${name}" is not there. Narrate the attempt and that it finds nothing; no luck for this.`,
};

/**
 * Before Generator (beforeAgent, after the Guard; stage T.8, ADR 71): the world leads the player.
 * One call (template `pregen`) reads the input against the world state and the Guard's warning:
 *
 * - **drafts**: people and fixtures the place makes sure (the bartender of a bar the player turns
 *   to) — the code takes only certain/likely, never items, at most MAX_DRAFTS, and makes them
 *   drafts (ADR 48): used in the turn, they stay; otherwise they are removed (Guard's afterAgent);
 * - **claims**: what the player says they have, notice or find that the world does not show — the
 *   code never makes them real: it tries the story's luck (recorded as `luck.rolled`, so the
 *   generator knows, ADR 47) and tells the master the outcome; a high threat gets no luck at all.
 *
 * Not for system commands or rejected inputs. A failed call: the turn goes on without it (ADR 36).
 */
export function beforeGeneratorMiddleware(options: BeforeGeneratorOptions) {
  return createMiddleware({
    name: 'BeforeGenerator',
    beforeAgent: async (state, runtime) => {
      const input = state.messages.at(-1);
      if (!input || sourceOf(input) !== 'player') return undefined;
      if (options.hints.decision?.decision === 'REJECT') return undefined;
      const answer = await askPregen(options, input.text, runtime.signal);
      if (!answer) return undefined;
      const sure: ExpectedGap[] = answer.drafts.filter((d) => SURE_LIKELIHOODS.includes(d.likelihood)).slice(0, MAX_DRAFTS);
      if (sure.length) await addDrafts(options.world, sure, options.onError);
      options.hints.checks = await judgeClaims(options, answer.claims.slice(0, MAX_CLAIMS));
      return undefined;
    },
  });
}

async function askPregen(options: BeforeGeneratorOptions, input: string, signal?: AbortSignal) {
  const { model, world, hints, timeoutMs = 15_000, onError } = options;
  const flagged = hints.decision && hints.decision.decision !== 'ALLOW' ? `${hints.decision.warningType}, threat ${hints.decision.threatLevel.toLowerCase()}: ${hints.warning}` : 'nothing flagged';
  try {
    const answer = await invokeWithin(
      model,
      [
        promptMessage(promptPart('pregen')),
        new HumanMessage(`# Current state\n${await renderWorldState(world)}\n\n# Input check\n${flagged}\n\n# Player input\n${input}`),
      ],
      { timeoutMs, signal },
    );
    return parseJsonAnswer(Pregen, answer.text);
  } catch (error) {
    if (signal?.aborted) throw error;
    onError?.(error);
    return undefined;
  }
}

/** Each unsupported claim: a luck roll in code (none at a high threat), the master's line. */
async function judgeClaims(options: BeforeGeneratorOptions, claims: z.infer<typeof Pregen>['claims']): Promise<string[]> {
  const { world, hints, luck = FAIR_LUCK, random = Math.random } = options;
  const lines: string[] = [];
  for (const claim of claims) {
    if (hints.decision?.threatLevel === 'HIGH') {
      lines.push(CLAIM_TEXT.refused(claim.name));
      continue;
    }
    const roll = rollLuck({ attempt: `${claim.name} (${claim.kind}, claimed by the player)`, likelihood: claim.likelihood, stakes: claim.stakes }, luck, random);
    const recorded = await world.actingAs('guard', () => world.recordLuck(roll));
    if (!recorded.ok) options.onError?.(new Error(recorded.error));
    lines.push(CLAIM_TEXT[roll.outcome](claim.name));
  }
  return lines;
}
