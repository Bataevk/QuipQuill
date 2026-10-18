/**
 * Conventions for the master's messages. Inputs carry their source in `additional_kwargs`
 * (kept in `model_messages`, not sent to the provider): what the player wrote, a system
 * command (session_start, stage 4.4), or the summary of older history (Context middleware).
 */
import { AIMessage, type BaseMessage, HumanMessage, ToolMessage } from '@langchain/core/messages';

/** A cache breakpoint for what changes call to call (5 minutes, the provider's default). */
export const CACHE = { type: 'ephemeral' } as const;
/**
 * For a prefix stable the whole session (system prompts): an hour, so a long pause of the player
 * does not make it written again (ADR 49). Costs 2× to write instead of 1.25×; must come before
 * any 5-minute breakpoint of the same request.
 */
export const CACHE_LONG = { type: 'ephemeral', ttl: '1h' } as const;

export type InputSource = 'player' | 'system' | 'summary';

const SOURCE = 'qq_source';
const TURN = 'qq_turn';

/** An input message; `turn` (the world's turn number) lets a failed turn be retried as itself. */
export function inputMessage(text: string, source: InputSource, turn?: number): HumanMessage {
  return new HumanMessage({
    content: text,
    additional_kwargs: { [SOURCE]: source, ...(turn === undefined ? {} : { [TURN]: turn }) },
  });
}

export function turnOf(message: BaseMessage): number | undefined {
  const turn = message.additional_kwargs?.[TURN];
  return typeof turn === 'number' ? turn : undefined;
}

export function sourceOf(message: BaseMessage): InputSource | undefined {
  if (!HumanMessage.isInstance(message)) return undefined;
  return (message.additional_kwargs?.[SOURCE] as InputSource | undefined) ?? 'player';
}

/** Index of the input that started the current turn (the last player or system input), or -1. */
export function turnStart(messages: BaseMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const source = sourceOf(messages[i]);
    if (source === 'player' || source === 'system') return i;
  }
  return -1;
}

/** The current turn: from the input that started it (the whole history when there is none). */
export function currentTurn(messages: BaseMessage[]): BaseMessage[] {
  return messages.slice(Math.max(turnStart(messages), 0));
}

/**
 * The messages of an `afterAgent` hook's state: its type leaves out the built-in messages,
 * though they are there.
 */
export function messagesOf(state: unknown): BaseMessage[] {
  return (state as { messages: BaseMessage[] }).messages;
}

// --- markers in the master's history --------------------------------------------------------

/** `additional_kwargs` key of an entity's own line written in the actor pack: `{ id, name }`. */
const ACTOR_LINE = 'qq_actor';
/** `additional_kwargs` key of a line that could not be written: the reason. */
const ACTOR_FAILED = 'qq_actor_failed';
/** `additional_kwargs` key of a tool result the ToolLimit wrote instead of running the call. */
const BLOCKED = 'qq_blocked';

export type EntityRef = { id: string; name: string };

/** An entity's line in the master's history (actor pack, ADR 43); with `failure`, why there is none. */
export function actorLine(entity: EntityRef, text: string, failure?: string): AIMessage {
  return new AIMessage({
    content: text,
    additional_kwargs: { [ACTOR_LINE]: entity, ...(failure === undefined ? {} : { [ACTOR_FAILED]: failure }) },
  });
}

/** The entity whose line this is (written in the actor pack), or undefined. */
export function actorOf(message: BaseMessage | undefined): EntityRef | undefined {
  if (!message || !AIMessage.isInstance(message)) return undefined;
  const ref = message.additional_kwargs?.[ACTOR_LINE] as EntityRef | undefined;
  return ref && typeof ref.id === 'string' ? ref : undefined;
}

/** Why an entity's line could not be written, or undefined. */
export function actorFailureOf(message: BaseMessage): string | undefined {
  const failed = message.additional_kwargs?.[ACTOR_FAILED];
  return failed ? String(failed) : undefined;
}

/** The Guard's reply in place of the master's (ADR 35); `reason` is for the log. */
export function guardReply(text: string, reason: string): AIMessage {
  return new AIMessage({ content: text, response_metadata: { guard: 'reject', reason } });
}

export function isGuardReply(message: BaseMessage): boolean {
  return AIMessage.isInstance(message) && message.response_metadata?.guard === 'reject';
}

/** The transcript's `extra` of a line that is the Guard's reply. */
export const GUARD_REPLY_EXTRA = { guard: 'reject' } as const;

/** A tool result written by ToolLimit in place of a call past the limit. */
export function blockedResult(call: { id?: string; name: string }, content: string): ToolMessage {
  return new ToolMessage({
    tool_call_id: call.id ?? '',
    name: call.name,
    status: 'error',
    content,
    additional_kwargs: { [BLOCKED]: true },
  });
}

export function isBlocked(message: BaseMessage | undefined): boolean {
  return !!message && ToolMessage.isInstance(message) && message.additional_kwargs?.[BLOCKED] === true;
}

/** The master's own replies with text in a turn, in order: entities' lines left out. */
export function masterReplies(turn: BaseMessage[]): AIMessage[] {
  return turn.filter((m): m is AIMessage => AIMessage.isInstance(m) && !!m.text.trim() && !actorOf(m));
}

/** Ids of tool calls that already have a result. */
export function answeredIds(messages: BaseMessage[]): Set<string> {
  return new Set(messages.flatMap((m) => (ToolMessage.isInstance(m) ? [m.tool_call_id] : [])));
}

/**
 * Rough token count: characters / 4, as the cloud estimates (protocol.md), plus tool call
 * arguments and a little per message. Good enough for thresholds far from the limits.
 */
export function estimateTokens(messages: BaseMessage[]): number {
  let chars = 0;
  for (const message of messages) {
    chars += message.text.length + 16;
    if (AIMessage.isInstance(message)) {
      for (const call of message.tool_calls ?? []) chars += call.name.length + JSON.stringify(call.args).length;
    }
  }
  return Math.ceil(chars / 4);
}

/**
 * Cache breakpoints for one call (ADR 46): on the last message with text, so the provider
 * caches everything up to it; when that last message is a tool result, also on the last input
 * or AI text before it (should the provider ignore a breakpoint in a tool result). Copies;
 * the history itself is not changed.
 */
export function withCacheBreakpoints(messages: BaseMessage[]): BaseMessage[] {
  const marked = [...messages];
  const last = marked.findLastIndex((m) => m.text.trim());
  if (last < 0) return marked;
  marked[last] = withBreakpoint(marked[last]);
  if (ToolMessage.isInstance(messages[last])) {
    const before = messages.findLastIndex((m, i) => i < last && !ToolMessage.isInstance(m) && m.text.trim());
    if (before >= 0) marked[before] = withBreakpoint(marked[before]);
  }
  return marked;
}

function withBreakpoint(message: BaseMessage): BaseMessage {
  const content = [{ type: 'text' as const, text: message.text, cache_control: CACHE }];
  const common = { id: message.id, content, additional_kwargs: message.additional_kwargs };
  if (ToolMessage.isInstance(message)) {
    return new ToolMessage({ ...common, tool_call_id: message.tool_call_id, name: message.name, status: message.status });
  }
  if (AIMessage.isInstance(message)) {
    return new AIMessage({
      ...common,
      tool_calls: message.tool_calls,
      usage_metadata: message.usage_metadata,
      response_metadata: message.response_metadata,
    });
  }
  if (HumanMessage.isInstance(message)) return new HumanMessage(common);
  return message;
}
