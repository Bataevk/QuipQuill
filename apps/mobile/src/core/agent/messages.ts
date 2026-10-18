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
