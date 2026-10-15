import { AIMessage, type BaseMessage, RemoveMessage } from '@langchain/core/messages';
import { createMiddleware } from 'langchain';

import { answeredIds } from '../messages';

/**
 * BugGuard (beforeAgent, first): drops tool calls that never got a result. The OS may kill the
 * app in the middle of a tool call; the history is saved step by step, so it can end with a
 * call and no answer — and the provider rejects a tool call without its result. An AI message
 * keeps the calls that were answered; one left with no calls and no text is removed.
 */
export function bugGuardMiddleware() {
  return createMiddleware({
    name: 'BugGuard',
    beforeAgent: (state) => {
      const fixes = repairDanglingToolCalls(state.messages);
      return fixes.length ? { messages: fixes } : undefined;
    },
  });
}

/** Replacements (same id) and removals that leave every tool call answered. */
export function repairDanglingToolCalls(messages: BaseMessage[]): BaseMessage[] {
  const answered = answeredIds(messages);
  const fixes: BaseMessage[] = [];
  for (const message of messages) {
    if (!AIMessage.isInstance(message) || !message.tool_calls?.length) continue;
    const kept = message.tool_calls.filter((call) => call.id && answered.has(call.id));
    if (kept.length === message.tool_calls.length) continue;
    if (!message.id) continue; // the reducer gives every message an id before hooks run
    if (kept.length === 0 && !message.text.trim()) {
      fixes.push(new RemoveMessage({ id: message.id }));
    } else {
      fixes.push(
        new AIMessage({
          id: message.id,
          content: message.content,
          tool_calls: kept,
          usage_metadata: message.usage_metadata,
          response_metadata: message.response_metadata,
        }),
      );
    }
  }
  return fixes;
}
