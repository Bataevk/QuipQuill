import { AIMessage, ToolMessage } from '@langchain/core/messages';
import { createMiddleware } from 'langchain';

import { answeredIds, turnStart } from '../messages';

/** Tool calls the master may run in one turn (agent.md). */
export const MAX_TOOL_CALLS_PER_TURN = 8;

export const TOOL_LIMIT_TEXT =
  'Error: no more actions this turn. Answer the player now, without calling tools.';

/**
 * ToolLimit (afterModel): at most `limit` tool calls run per turn. Calls past the limit get an
 * error result instead of running, and the model is asked again to answer in words; if it
 * still calls tools after that, the turn ends.
 */
export function toolLimitMiddleware(limit = MAX_TOOL_CALLS_PER_TURN) {
  return createMiddleware({
    name: 'ToolLimit',
    afterModel: {
      canJumpTo: ['model', 'end'],
      hook: (state) => {
        const { messages } = state;
        // The model's reply may already be followed by error results (InvalidToolCalls).
        const at = messages.findLastIndex((m) => AIMessage.isInstance(m));
        const last = messages[at];
        if (!last || !AIMessage.isInstance(last) || !last.tool_calls?.length) return undefined;
        if (messages.slice(at + 1).some((m) => !ToolMessage.isInstance(m))) return undefined;
        const turn = messages.slice(Math.max(turnStart(messages), 0), at);
        const results = turn.filter((m) => ToolMessage.isInstance(m));
        const blockedBefore = results.some((m) => m.additional_kwargs?.qq_blocked === true);
        const used = results.filter((m) => m.additional_kwargs?.qq_blocked !== true).length;
        const answered = answeredIds(messages);
        const pending = last.tool_calls.filter((call) => !call.id || !answered.has(call.id));
        const allowed = Math.max(0, limit - used);
        if (pending.length <= allowed) return undefined;
        const blocked = pending.slice(allowed).map(
          (call) =>
            new ToolMessage({
              tool_call_id: call.id ?? '',
              name: call.name,
              status: 'error',
              content: TOOL_LIMIT_TEXT,
              additional_kwargs: { qq_blocked: true },
            }),
        );
        if (allowed > 0) return { messages: blocked }; // the allowed calls still run
        return { messages: blocked, jumpTo: blockedBefore ? ('end' as const) : ('model' as const) };
      },
    },
  });
}
