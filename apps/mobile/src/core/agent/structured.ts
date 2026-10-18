/**
 * Calls with a JSON answer (stage R.1): the Guard, the generator, character adaptation and
 * entity notes. One way to bound a model call in time, to read its JSON answer and to write the
 * provider's `response_format`. The answer is read the same way with or without
 * `response_format`: a model that only writes JSON in its text works too.
 */
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import type { z } from 'zod';

/** The caller's signal joined with an explicit limit for one call. */
export function withTimeout(timeoutMs: number, signal?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

/** One model call with an explicit limit; aborted by `signal` too. */
export function invokeWithin(
  model: BaseChatModel,
  messages: BaseMessage[],
  { timeoutMs, signal }: { timeoutMs: number; signal?: AbortSignal },
) {
  return model.invoke(messages, { signal: withTimeout(timeoutMs, signal) });
}

/**
 * A JSON answer checked by `schema` (a ```json fence around it is tolerated). Throws when it is
 * not JSON or not of that shape; callers wrap the error in their own.
 */
export function parseJsonAnswer<T extends z.ZodType>(schema: T, text: string): z.output<T> {
  return schema.parse(JSON.parse(text.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '')));
}

/** OpenAI-style `response_format` with a strict JSON schema; the cloud passes it on as is (protocol.md). */
export function jsonSchemaFormat<S extends object>(name: string, schema: S) {
  return { type: 'json_schema', json_schema: { name, strict: true, schema } } as const;
}

/** An object for a strict schema: every property required, no others. */
export function strictObject<P extends Record<string, unknown>>(properties: P) {
  return { type: 'object', properties, required: Object.keys(properties), additionalProperties: false } as const;
}
