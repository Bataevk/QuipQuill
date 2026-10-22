import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, type BaseMessage, HumanMessage } from '@langchain/core/messages';
import { createMiddleware } from 'langchain';
import { z } from 'zod';

import type { DRAFT_TYPES } from '../../world/model';
import type { World } from '../../world/world';
import { promptMessage, promptPart } from '../cloud-prompt';
import { LIKELIHOODS } from '../luck';
import { actorOf, currentTurn, guardReply, messagesOf, sourceOf } from '../messages';
import { renderWorldState } from '../prompt';
import { miniGmReply, type MiniGmOptions } from '../mini-gm';
import { invokeWithin, parseJsonAnswer } from '../structured';

/** Gaps added as drafts: only sure ones, at most this many a turn. */
export const MAX_DRAFTS = 2;

/**
 * The Guard's three questions (ADR 70), answered by a decision model in the cloud: where the input
 * goes, how it writes the world, how serious it is. The questions themselves are in the cloud's
 * template `guard`.
 */
export const DECISIONS = ['ALLOW', 'PASS_TO_GM', 'REJECT'] as const;
export const WARNING_TYPES = ['NONE', 'STORY_HIJACKING', 'NPC_CONTROL', 'GHOST_ITEM', 'OTHER'] as const;
export const THREAT_LEVELS = ['LOW', 'MEDIUM', 'HIGH'] as const;

const choiceOf = <T extends readonly [string, ...string[]]>(values: T) => z.object({ choice: z.enum(values) });
/** The decision model's answers (`done.message`, JSON): a choice per question, probabilities beside. */
export const GuardAnswers = z.object({
  decision: choiceOf(DECISIONS),
  warning_type: choiceOf(WARNING_TYPES),
  threat_level: choiceOf(THREAT_LEVELS),
});

export interface GuardDecision {
  decision: (typeof DECISIONS)[number];
  warningType: (typeof WARNING_TYPES)[number];
  threatLevel: (typeof THREAT_LEVELS)[number];
}

/** Ways of writing the world (ТЗ 4.4): an input that does one of them goes to the master flagged. */
const WRITES_THE_WORLD: readonly GuardDecision['warningType'][] = ['STORY_HIJACKING', 'NPC_CONTROL', 'GHOST_ITEM'];

/**
 * The answers made consistent, on the strict side (owner: the Guard is stricter now): an input
 * allowed although it writes the world goes to the master flagged; a flagged or rejected input
 * always names a kind; an allowed one has none.
 */
export function guardDecision(answers: z.infer<typeof GuardAnswers>): GuardDecision {
  let decision = answers.decision.choice;
  let warningType = answers.warning_type.choice;
  const threatLevel = answers.threat_level.choice;
  if (decision === 'ALLOW' && WRITES_THE_WORLD.includes(warningType)) decision = 'PASS_TO_GM';
  if (decision === 'ALLOW') warningType = 'NONE';
  else if (warningType === 'NONE') warningType = 'OTHER';
  return { decision, warningType, threatLevel };
}

/**
 * What the master reads in the world state when the input is flagged (`DANGER`, this turn only):
 * a fixed line per kind, not the model's words — the decision model writes none (ADR 70).
 */
export const WARNING_TEXT: Record<Exclude<GuardDecision['warningType'], 'NONE'>, string> = {
  STORY_HIJACKING:
    'the player declares what happens instead of attempting it. Narrate only what their character actually tries; the world decides the outcome.',
  NPC_CONTROL:
    'the player speaks, decides or acts for someone else. People answer for themselves: ignore what the player put in their mouths and let them react on their own.',
  GHOST_ITEM:
    'the player claims something the world state does not show. It is not there: narrate the attempt and that it finds nothing.',
  OTHER: 'the input breaks the rules of play. Take only what the player character attempts.',
};

export function warningText({ warningType, threatLevel }: GuardDecision): string {
  return warningType === 'NONE' ? '' : `${WARNING_TEXT[warningType]} (threat: ${threatLevel.toLowerCase()})`;
}

/**
 * The answer to a rejected input when Mini-GM cannot write one (stage T.6): a short line in the
 * story's voice. Picked by the input, so a retried turn says the same.
 */
export const REJECT_REPLIES = [
  'The story pauses, unimpressed: that is not something your character can do here. What do they do instead?',
  'The quill refuses to write that down. Back to the story: what does your character do?',
  'Somewhere, a narrator raises an eyebrow. That belongs outside this tale; your character is still waiting.',
];

export function rejectReply(input: string): string {
  let hash = 0;
  for (const char of input) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return REJECT_REPLIES[hash % REJECT_REPLIES.length];
}

/** A gap of the place the input turns to: a person at work there or a fixture the world does not show. */
export interface ExpectedGap {
  name: string;
  /** People and fixtures of the place, never items (ADR 48). */
  type: Lowercase<(typeof DRAFT_TYPES)[number]>;
  description: string;
  likelihood: (typeof LIKELIHOODS)[number];
}

/**
 * What the input check left for this turn, reset at every input. `decision`: the Guard's (ADR 70);
 * `warning`: on a flagged input, the master's DANGER line. `expected`: the place's sure gaps — the
 * master sees them in the world state, the generator after the turn (ADR 54); the decision model
 * names none, so they stay empty until Before Generator fills them (stage T.8).
 */
export interface GuardHints {
  expected: ExpectedGap[];
  warning: string;
  decision?: GuardDecision;
  /** Before Generator's word on the player's unsupported claims, for the master (stage T.8). */
  checks: string[];
}

export const createGuardHints = (): GuardHints => ({ expected: [], warning: '', checks: [] });

export interface GuardOptions {
  /** The decision model (QueueChatModel with purpose 'guard', ADR 70). */
  model: BaseChatModel;
  world: World;
  /** `meta.world_constraints`: era, peoples, magic and technology. */
  worldConstraints: string;
  timeoutMs?: number;
  /** Called on a failed check (the input passes anyway). */
  onError?: (error: unknown) => void;
  /** Where the turn's decision and warning go; without it they are dropped. */
  hints?: GuardHints;
  /** Who answers a rejected input (stage T.6); without it, or when it fails, a fixed line. */
  miniGm?: MiniGmOptions;
}

/**
 * Guard (beforeAgent, after BugGuard): one decision-model call routes the player's input
 * (agent.md, ADR 70). `ALLOW` — on to the master; `PASS_TO_GM` — the input writes the world (an
 * outcome, someone else's words, a thing that is not there): on to the master with a DANGER line
 * for this turn; `REJECT` — not playing the story (talking to the AI, injections, unrelated work):
 * Mini-GM's short line is the turn's answer and the master is not called (ADR 35, stage T.6). System commands and
 * summaries are not checked. A failed check — network, limit, timeout, malformed answer — lets
 * the input through as `ALLOW`: the master's tools still enforce the world (ADR 36).
 * After the turn (afterAgent) a draft the turn used stays, the rest are removed (ADR 48).
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
        const decision = await checkPlayerInput(options, state.messages, runtime.signal);
        if (!decision) return undefined;
        if (options.hints) Object.assign(options.hints, { decision, warning: warningText(decision) });
        if (decision.decision !== 'REJECT') return undefined;
        const reply = await answerRejected(options, input.text, lastNarration(state.messages.slice(0, -1)), runtime.signal);
        return {
          messages: [guardReply(reply, `${decision.warningType} ${decision.threatLevel}`)],
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

/** Mini-GM's line for a rejected input, or the fixed one when there is none (stage T.6). */
async function answerRejected(
  options: GuardOptions,
  input: string,
  narration: string,
  signal?: AbortSignal,
): Promise<string> {
  if (!options.miniGm) return rejectReply(input);
  try {
    return await miniGmReply(options.miniGm, input, narration, signal);
  } catch (error) {
    if (signal?.aborted) throw error;
    options.onError?.(error);
    return rejectReply(input);
  }
}

/** Sure gaps as drafts in the current location (a refused one is skipped); Before Generator (T.8). */
export async function addDrafts(world: World, sure: ExpectedGap[], onError?: (error: unknown) => void) {
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

/** The master's last words before the input (not an entity's line), for Mini-GM. */
function lastNarration(history: BaseMessage[]): string {
  const last = history.findLast((m) => AIMessage.isInstance(m) && !actorOf(m) && m.text.trim());
  return last?.text ?? '';
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

/**
 * The decision on the last message (the player's input), or undefined when the check failed (then
 * the input passes). Exported for the live check of the Guard (demo.live.test.ts).
 */
export async function checkPlayerInput(
  { model, world, worldConstraints, timeoutMs = 15_000, onError }: GuardOptions,
  messages: BaseMessage[],
  signal?: AbortSignal,
): Promise<GuardDecision | undefined> {
  const input = messages.at(-1)!;
  const entries = guardConversation(messages.slice(0, -1));
  // No cache marks: the decision model is not cached (ADR 70).
  const conversation = entries.length ? [`# Recent conversation (oldest first)\n${entries.join('\n')}`] : [];
  const now = `# Current state\n${await renderWorldState(world)}\n\n# Player input\n${input.text}`;
  try {
    const answer = await invokeWithin(
      model,
      [
        // The cloud's template `guard` (ADR 57, 70): the opening, the world rules and the questions.
        promptMessage(promptPart('guard', { WORLD_RULES: worldConstraints.trim() || 'none given' })),
        new HumanMessage([...conversation, now].join('\n\n')),
      ],
      { timeoutMs, signal },
    );
    return guardDecision(parseJsonAnswer(GuardAnswers, answer.text));
  } catch (error) {
    if (signal?.aborted) throw error; // the turn itself was cancelled
    onError?.(error);
    return undefined;
  }
}
