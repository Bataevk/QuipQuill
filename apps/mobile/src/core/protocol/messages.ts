/**
 * Client ↔ cloud wire format, v1. Spec: docs/architecture/protocol.md; server side:
 * cloud/src/quipquill_cloud/mq/schemas.py. Keep the two in sync.
 */
export const PROTOCOL_VERSION = 1;

export type Purpose = 'gm' | 'guard' | 'npc' | 'summary' | 'character' | 'generator';

// --- OpenAI-format chat payload (passed to the provider as is) --------------------------------

export interface TextPart {
  type: 'text';
  text: string;
  /** Prompt cache breakpoint (ADR 20, 46); `ttl: '1h'` for prefixes stable all session (ADR 49). */
  cache_control?: { type: 'ephemeral'; ttl?: '1h' };
}

/**
 * A prompt template the cloud fills in (ADR 57): the app names it and gives its variables, the
 * worker puts the text there (`{{VAR}}` → value) and keeps `cache_control`. The app never ships
 * the prompts' text; templates: `prompts/` at the repository root, in production CI variables.
 */
export interface PromptPart {
  type: 'prompt';
  name: PromptName;
  vars: Record<string, string>;
  cache_control?: TextPart['cache_control'];
}

export type PromptName = 'gm' | 'actor' | 'guard' | 'generator' | 'character' | 'summary' | 'note_entity';

export type ContentPart = TextPart | PromptPart;

export interface WireToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

export type WireMessage =
  | { role: 'system'; content: string | ContentPart[] }
  | { role: 'user'; content: string | ContentPart[] }
  | { role: 'assistant'; content: string | TextPart[] | null; tool_calls?: WireToolCall[] }
  | { role: 'tool'; content: string | TextPart[]; tool_call_id: string };

export interface WireTool {
  type: 'function';
  function: { name: string; description?: string; parameters: Record<string, unknown> };
}

// --- client → server ---------------------------------------------------------------------------

export interface AuthMessage {
  v: 1;
  type: 'auth';
  token: string;
}

/** Model classes the client may ask for; which model and providers serve one is the server's choice. */
export type ModelClass = 'basic';

export interface LlmRequest {
  v: 1;
  type: 'llm.request';
  requestId: string;
  sessionId: string;
  /** Where the UI draws the stream: 'gm' | 'npc:<id>' | 'guard' | 'system'. */
  channel: string;
  purpose: Purpose;
  /** The model class (ADR 50): the server maps it to a model and providers; default 'basic'. */
  model?: ModelClass;
  messages: WireMessage[];
  tools?: WireTool[];
  responseFormat?: Record<string, unknown>;
  meta?: Record<string, unknown>;
}

export type ClientMessage =
  | AuthMessage
  | LlmRequest
  | { v: 1; type: 'resume'; sessionId: string }
  | { v: 1; type: 'cancel'; requestId: string }
  | { v: 1; type: 'ping' };

// --- server → client ---------------------------------------------------------------------------

export interface Usage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite?: number;
  costUsd: number;
}

interface Routed {
  v: 1;
  requestId: string;
  sessionId: string;
  channel?: string;
  meta?: Record<string, unknown>;
}

export type FinishReason = 'stop' | 'tool_calls' | 'length' | 'cancelled' | string;

export type StatusEvent = Routed & { type: 'status'; status: 'queued' | 'thinking' };
export type DeltaEvent = Routed & { type: 'delta'; seq: number; text: string };
export type ToolCallEvent = Routed & { type: 'tool_call'; call: WireToolCall };
export type DoneEvent = Routed & {
  type: 'done';
  /** The agent's source of truth: the whole assistant message with complete tool calls. */
  message: { role: 'assistant'; content: string | null; tool_calls?: WireToolCall[] };
  finishReason: FinishReason;
  usage: Usage;
};
export type ErrorEvent = Partial<Routed> & {
  v: 1;
  type: 'error';
  code: string;
  statusKey: string;
  retryable: boolean;
};
/** After `resume`: the whole text so far — replaces what the UI shows. */
export type ReplayEvent = Routed & { type: 'replay'; text: string };

export type ServerEvent =
  | { v: 1; type: 'ready' }
  | { v: 1; type: 'heartbeat' }
  | StatusEvent
  | DeltaEvent
  | ToolCallEvent
  | DoneEvent
  | ErrorEvent
  | ReplayEvent;

const SERVER_TYPES = new Set([
  'ready',
  'heartbeat',
  'status',
  'delta',
  'tool_call',
  'done',
  'error',
  'replay',
]);

/** Parses a server frame; undefined for anything that is not a v1 event we know. */
export function parseServerEvent(raw: string): ServerEvent | undefined {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return undefined;
  }
  if (typeof data !== 'object' || data === null) return undefined;
  const event = data as { v?: unknown; type?: unknown };
  if (event.v !== PROTOCOL_VERSION || typeof event.type !== 'string') return undefined;
  return SERVER_TYPES.has(event.type) ? (data as ServerEvent) : undefined;
}

/** Close codes the server uses (protocol.md#подключение). */
export const CLOSE_UNAUTHORIZED = 4401;
export const CLOSE_TRY_AGAIN_LATER = 1013;
