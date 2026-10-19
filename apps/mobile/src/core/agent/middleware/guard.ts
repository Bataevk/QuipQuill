import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, type BaseMessage, HumanMessage } from '@langchain/core/messages';
import { createMiddleware } from 'langchain';
import { z } from 'zod';

import { DRAFT_TYPES, lowerTypes } from '../../world/model';
import type { World } from '../../world/world';
import { promptMessage, promptPart } from '../cloud-prompt';
import { LIKELIHOODS, SURE_LIKELIHOODS } from '../luck';
import { actorOf, CACHE, CACHE_LONG, currentTurn, guardReply, messagesOf, sourceOf } from '../messages';
import { renderWorldState } from '../prompt';
import { invokeWithin, jsonSchemaFormat, parseJsonAnswer, strictObject } from '../structured';

/** What the Guard may add before the master: people and fixtures of the place, never items (ADR 48). */
const EXPECTED_TYPES = lowerTypes(DRAFT_TYPES);
/** Gaps the Guard adds as drafts: only sure ones, at most this many a turn. */
export const MAX_DRAFTS = 2;


/** pass; warn — pass with a warning for the master; reject — off the story, the Guard answers (ADR 56). */
export const VERDICTS = ['pass', 'warn', 'reject'] as const;

/** OpenAI-style `response_format` the cloud passes to the provider as is (protocol.md). */
export const GUARD_RESPONSE_FORMAT = jsonSchemaFormat(
  'guard_verdict',
  strictObject({
    verdict: { type: 'string', enum: [...VERDICTS] },
    reason: { type: 'string' },
    reply: { type: 'string' },
    expected: {
      type: 'array',
      items: strictObject({
        name: { type: 'string' },
        type: { type: 'string', enum: [...EXPECTED_TYPES] },
        description: { type: 'string' },
        likelihood: { type: 'string', enum: [...LIKELIHOODS] },
      }),
    },
  }),
);

export const Verdict = z.object({
  verdict: z.enum(VERDICTS),
  reason: z.string().default(''),
  reply: z.string().default(''),
  expected: z
    .array(
      z.object({
        name: z.string().trim().min(1),
        type: z.enum(EXPECTED_TYPES),
        description: z.string().default(''),
        likelihood: z.enum(LIKELIHOODS),
      }),
    )
    .default([]),
});
export type GuardVerdict = z.infer<typeof Verdict>;
/** A gap the Guard expects in this place: a person at work there or a fixture the world does not show. */
export type ExpectedGap = GuardVerdict['expected'][number];

/**
 * What the Guard saw in the turn it checked, reset at every input. `expected`: the sure gaps of
 * the place (ADR 54) — the master sees them in the world state and may make one real if the
 * moment needs it; the generator adds what fits after the turn. `warning`: on `warn`, why the
 * input is not allowed (ADR 56) — the master sees it in the world state and decides.
 */
export interface GuardHints {
  expected: ExpectedGap[];
  warning: string;
}

export const createGuardHints = (): GuardHints => ({ expected: [], warning: '' });

export interface GuardOptions {
  /** A cheap model (QueueChatModel with purpose 'guard' and GUARD_RESPONSE_FORMAT). */
  model: BaseChatModel;
  world: World;
  /** `meta.world_constraints`: era, peoples, magic and technology. */
  worldConstraints: string;
  timeoutMs?: number;
  /** Called on a failed check (the input passes anyway). */
  onError?: (error: unknown) => void;
  /** Where the turn's recommendations go (ADR 54); without it they are dropped. */
  hints?: GuardHints;
  /**
   * The Guard writes its gaps into the world as drafts (ADR 48). Off since ADR 54: the Guard only
   * recommends; the code stays for a later decision.
   */
  drafts?: boolean;
}

/**
 * Guard (beforeAgent, after BugGuard): one cheap call checks the player's input (agent.md). It is
 * only a check (ADR 56): `pass`; `warn` — the input tries to write the world, the master gets it
 * with the Guard's reason as a DANGER warning and decides; `reject` — the input is not about the
 * story at all, the Guard's own reply is the turn's answer and the master is not called (ADR 35). System commands and summaries are not checked. A failed check — network,
 * limit, timeout, malformed answer — lets the input through: the master's tools still enforce
 * the world, and the player should not get stuck on the cheap check (owner, stage 4.2).
 *
 * On `pass` and `warn` the sure gaps it saw (a bartender in a bar the world does not show), at most
 * MAX_DRAFTS, are recommendations (ADR 54): `hints` for the master's world state and the generator.
 * With `drafts` (ADR 48, off) they become drafts in the world instead; after the turn (afterAgent)
 * a draft the turn used stays, the rest are removed.
 */
export function guardMiddleware(options: GuardOptions) {
  return createMiddleware({
    name: 'Guard',
    beforeAgent: {
      canJumpTo: ['end'],
      hook: async (state, runtime) => {
        if (options.hints) Object.assign(options.hints, createGuardHints());
        const input = state.messages.at(-1);
        if (!input || sourceOf(input) !== 'player') return undefined;
        const verdict = await checkInput(options, state.messages, runtime.signal);
        if (verdict?.verdict !== 'reject' || !verdict.reply.trim()) {
          const sure = (verdict?.expected ?? []).filter((e) => SURE_LIKELIHOODS.includes(e.likelihood)).slice(0, MAX_DRAFTS);
          if (options.drafts) await addDrafts(options.world, sure, options.onError);
          else if (options.hints) options.hints.expected = sure;
          if (verdict?.verdict === 'warn' && options.hints) {
            options.hints.warning = verdict.reason.trim() || 'the input claims more than the world allows.';
          }
          return undefined;
        }
        return {
          messages: [guardReply(verdict.reply.trim(), verdict.reason)],
          jumpTo: 'end' as const,
        };
      },
    },
    afterAgent: async (state) => {
      const narration = currentTurn(messagesOf(state)).flatMap((m) => (AIMessage.isInstance(m) ? [m.text] : [])).join('\n');
      await settleDrafts(options.world, narration, options.onError);
    },
  });
}

/** The Guard's sure gaps as drafts in the current location (a refused one is skipped). */
async function addDrafts(world: World, sure: ExpectedGap[], onError?: (error: unknown) => void) {
  await world.actingAs('guard', async () => {
    for (const e of sure) {
      const added = await world.setEntity({ name: e.name, type: e.type, description: e.description, draft: true });
      if (!added.ok) onError?.(new Error(`draft ${e.name}: ${added.error}`));
    }
  });
}

/**
 * After the turn: a draft stays if the turn used it — an event names it (it reacted, was moved,
 * tied, edited) or the master's narration mentions it; otherwise it is removed from the world.
 */
export async function settleDrafts(world: World, narration: string, onError?: (error: unknown) => void) {
  const drafts = await world.drafts();
  if (!drafts.length) return;
  const events = await world.turnEvents();
  const text = narration.toLowerCase();
  await world.actingAs('guard', async () => {
    for (const draft of drafts) {
      const named = events.some(
        (e) => e.kind !== 'entity.added' && JSON.stringify(e.payload).includes(`"${draft.id}"`),
      );
      const used = named || text.includes(draft.name.toLowerCase());
      const done = used ? await world.confirmDraft(draft.id) : await world.withdrawDraft(draft.id);
      if (!done.ok) onError?.(new Error(done.error));
    }
  });
}

/** Conversation entries the Guard reads (ADR 49): at least this many, fewer than twice as many. */
export const GUARD_HISTORY = 10;

/**
 * The recent conversation for the Guard: the player's inputs, the master's narration and
 * entities' lines (no tool calls), oldest first. The window's start moves in steps of
 * GUARD_HISTORY, not with every turn, so the cached prefix holds for several turns.
 */
export function guardConversation(history: BaseMessage[]): string[] {
  const entries = history.flatMap(entryOf);
  if (entries.length <= GUARD_HISTORY) return entries;
  return entries.slice(Math.floor((entries.length - GUARD_HISTORY) / GUARD_HISTORY) * GUARD_HISTORY);
}

function entryOf(message: BaseMessage): string[] {
  const text = message.text.trim();
  if (!text) return [];
  if (HumanMessage.isInstance(message)) {
    const source = sourceOf(message);
    if (source === 'player') return [`Player: ${text}`];
    return source === 'summary' ? [`Earlier: ${text}`] : [];
  }
  if (!AIMessage.isInstance(message)) return [];
  const actor = actorOf(message);
  return [actor ? `${actor.name}: ${text}` : `Game Master: ${text}`];
}

/** The verdict, or undefined when the check failed (then the input passes). */
async function checkInput(
  { model, world, worldConstraints, timeoutMs = 15_000, onError }: GuardOptions,
  messages: BaseMessage[],
  signal: AbortSignal | undefined,
): Promise<GuardVerdict | undefined> {
  const input = messages.at(-1)!;
  const entries = guardConversation(messages.slice(0, -1));
  // System (an hour) → each entry its own block, a breakpoint on the last → what changes now.
  const conversation = entries.length
    ? [
        { type: 'text' as const, text: '# Recent conversation (oldest first)\n' },
        ...entries.map((entry, i) => ({
          type: 'text' as const,
          text: `${entry}\n`,
          ...(i === entries.length - 1 ? { cache_control: CACHE } : {}),
        })),
      ]
    : [];
  const now = `\n# Current state\n${await renderWorldState(world)}\n\n# Player input\n${input.text}`;
  try {
    const answer = await invokeWithin(
      model,
      [
        // The cloud's template `guard` (ADR 56, 57) with the story's world rules.
        promptMessage(promptPart('guard', { WORLD_RULES: worldConstraints.trim() || 'none given' }, CACHE_LONG)),
        new HumanMessage({ content: [...conversation, { type: 'text', text: now }] }),
      ],
      { timeoutMs, signal },
    );
    return parseJsonAnswer(Verdict, answer.text);
  } catch (error) {
    if (signal?.aborted) throw error; // the turn itself was cancelled
    onError?.(error);
    return undefined;
  }
}
