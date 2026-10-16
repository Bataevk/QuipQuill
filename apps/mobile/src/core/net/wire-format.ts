/** LangChain messages ↔ the OpenAI-format messages the cloud passes to the provider. */
import { AIMessage, type BaseMessage, type ToolMessage } from '@langchain/core/messages';

import type { DoneEvent, TextPart, WireMessage, WireToolCall } from '../protocol/messages';

type ContentBlock = { type?: string; text?: string; cache_control?: { type: 'ephemeral' } };

export function toWireMessages(messages: BaseMessage[]): WireMessage[] {
  return messages.map(toWireMessage);
}

function toWireMessage(message: BaseMessage): WireMessage {
  switch (message.getType()) {
    case 'system':
      return { role: 'system', content: toParts(message.content) };
    case 'human':
      return { role: 'user', content: toParts(message.content) };
    case 'ai': {
      const ai = message as AIMessage;
      const toolCalls: WireToolCall[] = (ai.tool_calls ?? []).map((call) => ({
        id: call.id ?? '',
        type: 'function',
        function: { name: call.name, arguments: JSON.stringify(call.args ?? {}) },
      }));
      const text = cached(ai.content) ? toParts(ai.content) : plainText(ai.content);
      return toolCalls.length > 0
        ? { role: 'assistant', content: text || null, tool_calls: toolCalls }
        : { role: 'assistant', content: text };
    }
    case 'tool': {
      const tool = message as ToolMessage;
      const content = cached(tool.content) ? toParts(tool.content) : plainText(tool.content);
      return { role: 'tool', content, tool_call_id: tool.tool_call_id };
    }
    default:
      throw new Error(`Unsupported message type: ${message.getType()}`);
  }
}

/** Keeps text blocks (and their cache_control breakpoints); plain strings stay strings. */
function toParts(content: BaseMessage['content']): string | TextPart[] {
  if (typeof content === 'string') return content;
  return (content as ContentBlock[])
    .filter((block) => block.type === 'text' && typeof block.text === 'string')
    .map((block) => ({
      type: 'text' as const,
      text: block.text as string,
      ...(block.cache_control ? { cache_control: block.cache_control } : {}),
    }));
}

/** Whether the content carries a cache breakpoint (then its text blocks are kept as parts). */
function cached(content: BaseMessage['content']): boolean {
  return typeof content !== 'string' && (content as ContentBlock[]).some((block) => block.cache_control);
}

function plainText(content: BaseMessage['content']): string {
  if (typeof content === 'string') return content;
  return (content as ContentBlock[])
    .filter((block) => block.type === 'text')
    .map((block) => block.text ?? '')
    .join('');
}

/** `done` → AIMessage: tool call arguments parsed; unparsable ones become invalid_tool_calls. */
export function fromDone(done: DoneEvent): AIMessage {
  const toolCalls: { id: string; name: string; args: Record<string, unknown>; type: 'tool_call' }[] = [];
  const invalid: { id: string; name: string; args: string; error: string; type: 'invalid_tool_call' }[] = [];
  for (const call of done.message.tool_calls ?? []) {
    try {
      const args = JSON.parse(call.function.arguments || '{}') as unknown;
      if (typeof args !== 'object' || args === null || Array.isArray(args)) throw new Error('not an object');
      toolCalls.push({ id: call.id, name: call.function.name, args: args as Record<string, unknown>, type: 'tool_call' });
    } catch (error) {
      invalid.push({
        id: call.id,
        name: call.function.name,
        args: call.function.arguments,
        error: `Arguments are not a JSON object: ${(error as Error).message}`,
        type: 'invalid_tool_call',
      });
    }
  }
  const { usage } = done;
  return new AIMessage({
    content: done.message.content ?? '',
    tool_calls: toolCalls,
    invalid_tool_calls: invalid,
    usage_metadata: {
      input_tokens: usage.input,
      output_tokens: usage.output,
      total_tokens: usage.input + usage.output,
      input_token_details: { cache_read: usage.cacheRead, cache_creation: usage.cacheWrite ?? 0 },
    },
    response_metadata: {
      finish_reason: done.finishReason,
      cost_usd: usage.costUsd,
      request_id: done.requestId,
    },
  });
}
