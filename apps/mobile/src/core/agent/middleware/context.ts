import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import {
  AIMessage,
  type BaseMessage,
  HumanMessage,
  RemoveMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { REMOVE_ALL_MESSAGES } from '@langchain/langgraph';
import { createMiddleware } from 'langchain';

import { estimateTokens, inputMessage, sourceOf, turnStart } from '../messages';

export const SUMMARY_PROMPT = `You keep the Game Master's memory of an interactive story. Below is the older part of the conversation between the player and the Game Master, possibly starting with an earlier summary. Rewrite all of it as one summary that replaces it.

Keep what the Game Master needs to continue consistently: what happened and where, in order; what the player did, said, promised or learned; what people said and how they feel about the player; clues, secrets revealed, debts, deals and unresolved threads. Keep names exactly as written. Skip tool mechanics and ids; the world state (places, items, notes) is kept separately.

Write in the past tense, plain prose, at most 400 words. Answer with the summary only.`;

export const SUMMARY_PREFIX = 'Summary of the story so far:';
export const TRIMMED_NOTE =
  '[Earlier conversation was dropped to save space. Rely on the world state, the notes and the recent events.]';

export interface ContextOptions {
  /** A cheap model for summaries (QueueChatModel with purpose 'summary'). */
  model?: BaseChatModel;
  /** Summarize once the history reaches this many tokens (~30K, agent.md). */
  triggerTokens?: number;
  /** The newest part kept as is after summarizing. */
  keepTokens?: number;
  /**
   * The history never exceeds this, summary or not; leaves room for the system prompt and
   * tools under the 100K price step of Haiku (ADR 9).
   */
  maxTokens?: number;
  timeoutMs?: number;
  retries?: number;
  onError?: (error: unknown, attempt: number) => void;
  /** Put in place of dropped history (default TRIMMED_NOTE, written for the master). */
  trimmedNote?: string;
}

const DEFAULTS = { triggerTokens: 30_000, keepTokens: 8_000, maxTokens: 85_000, timeoutMs: 25_000, retries: 2 };
/** A tool result longer than this is cut when nothing else is left to drop. */
const MAX_TOOL_CHARS = 4_000;

/**
 * Context (beforeModel): keeps the master's history small. Past the trigger, the older part
 * becomes one summary written by a cheap model (with a timeout and retries); if that fails,
 * the older part is dropped with a note instead. Cuts never split a tool call from its result,
 * and the current turn is kept whole when possible. Above `maxTokens` — whatever the summary
 * did — older turns go, then long tool results are shortened, then the current turn is cut. The change replaces the
 * whole history (RemoveMessage(REMOVE_ALL)); runTurn rewrites `model_messages` to match.
 * Idea from AI Samba's SafeSummarizationMiddleware (ADR 13: ideas, not code).
 */
export function contextMiddleware(options: ContextOptions = {}) {
  const o = { ...DEFAULTS, ...options };
  return createMiddleware({
    name: 'Context',
    beforeModel: async (state, runtime) => {
      const next = await compactHistory(state.messages, o, runtime.signal);
      return next ? { messages: [new RemoveMessage({ id: REMOVE_ALL_MESSAGES }), ...next] } : undefined;
    },
  });
}

/** The new history, or undefined when nothing has to change. */
export async function compactHistory(
  messages: BaseMessage[],
  options: ContextOptions = {},
  signal?: AbortSignal,
): Promise<BaseMessage[] | undefined> {
  const o = { ...DEFAULTS, ...options };
  let result: BaseMessage[] | undefined;
  if (estimateTokens(messages) >= o.triggerTokens) {
    const limit = Math.max(turnStart(messages), 0); // keep the current turn whole
    const cut = Math.min(cutForTail(messages, o.keepTokens), safeCutAtOrBefore(messages, limit));
    if (cut > 0) {
      const older = messages.slice(0, cut);
      const summary = await summarize(older, o, signal);
      const head = summary ? `${SUMMARY_PREFIX}\n${summary}` : (o.trimmedNote ?? TRIMMED_NOTE);
      result = [inputMessage(head, 'summary'), ...messages.slice(cut)];
    }
  }
  const capped = enforceCeiling(result ?? messages, o.maxTokens, o.trimmedNote ?? TRIMMED_NOTE);
  return capped ?? result;
}

/** Summary text, or undefined after all attempts failed. */
async function summarize(
  older: BaseMessage[],
  o: typeof DEFAULTS & ContextOptions,
  signal?: AbortSignal,
): Promise<string | undefined> {
  if (!o.model) return undefined;
  const transcript = older.map(line).filter(Boolean).join('\n');
  for (let attempt = 1; attempt <= o.retries; attempt++) {
    const timeout = AbortSignal.timeout(o.timeoutMs);
    try {
      const answer = await o.model.invoke([new SystemMessage(SUMMARY_PROMPT), new HumanMessage(transcript)], {
        signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      });
      const text = answer.text.trim();
      if (text) return text;
      throw new Error('empty summary');
    } catch (error) {
      if (signal?.aborted) throw error;
      o.onError?.(error, attempt);
    }
  }
  return undefined;
}

/** One message as a transcript line for the summarizer. */
function line(message: BaseMessage): string {
  if (sourceOf(message) === 'summary') return message.text;
  if (HumanMessage.isInstance(message)) {
    return sourceOf(message) === 'system' ? `[System] ${message.text}` : `Player: ${message.text}`;
  }
  if (AIMessage.isInstance(message)) {
    const calls = (message.tool_calls ?? []).map((c) => `[GM action] ${c.name} ${JSON.stringify(c.args)}`);
    return [message.text.trim() && `GM: ${message.text.trim()}`, ...calls].filter(Boolean).join('\n');
  }
  if (ToolMessage.isInstance(message)) return `[Result] ${message.text.slice(0, 500)}`;
  return '';
}

/** Can the history start at `index` without orphaning a tool result? */
function isSafeCut(messages: BaseMessage[], index: number): boolean {
  if (index <= 0 || index >= messages.length) return index >= 0 && index <= messages.length;
  return !ToolMessage.isInstance(messages[index]);
}

function safeCutAtOrBefore(messages: BaseMessage[], index: number): number {
  for (let i = Math.min(index, messages.length); i > 0; i--) if (isSafeCut(messages, i)) return i;
  return 0;
}

/** The smallest safe cut whose tail fits in `tokens` (0 if everything fits). */
function cutForTail(messages: BaseMessage[], tokens: number): number {
  let total = 0;
  let i = messages.length;
  while (i > 0 && total + estimateTokens([messages[i - 1]]) <= tokens) {
    i -= 1;
    total += estimateTokens([messages[i]]);
  }
  for (let j = i; j < messages.length; j++) if (isSafeCut(messages, j)) return j;
  return messages.length;
}

/**
 * Makes the history fit: drops older turns (keeping the current one whole), then shortens long
 * tool results, and only then cuts into the current turn.
 */
function enforceCeiling(messages: BaseMessage[], maxTokens: number, noteText: string): BaseMessage[] | undefined {
  if (estimateTokens(messages) <= maxTokens) return undefined;
  const note = inputMessage(noteText, 'summary');
  const room = maxTokens - estimateTokens([note]);
  const cut = Math.min(cutForTail(messages, room), safeCutAtOrBefore(messages, Math.max(turnStart(messages), 0)));
  let tail = messages.slice(cut);
  if (estimateTokens(tail) > room) tail = tail.map(shorten);
  let more = 0;
  if (estimateTokens(tail) > room) {
    more = cutForTail(tail, room);
    if (more >= tail.length) more = safeCutAtOrBefore(tail, tail.length - 1);
  }
  tail = tail.slice(more);
  return cut + more > 0 ? [note, ...tail] : tail;
}

function shorten(message: BaseMessage): BaseMessage {
  if (!ToolMessage.isInstance(message) || message.text.length <= MAX_TOOL_CHARS) return message;
  return new ToolMessage({
    id: message.id,
    tool_call_id: message.tool_call_id,
    name: message.name,
    status: message.status,
    additional_kwargs: message.additional_kwargs,
    content: `${message.text.slice(0, MAX_TOOL_CHARS)}… [cut]`,
  });
}
