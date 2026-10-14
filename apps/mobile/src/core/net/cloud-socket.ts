import {
  CLOSE_UNAUTHORIZED,
  parseServerEvent,
  type ClientMessage,
  type DoneEvent,
  type LlmRequest,
  type ServerEvent,
} from '../protocol/messages';
import { AuthRequiredError, type AuthSession } from './auth';

/** A failed generation (protocol `error` event): `code` per protocol.md, `retryable` hint. */
export class LlmError extends Error {
  readonly code: string;
  readonly retryable: boolean;

  constructor(code: string, retryable: boolean) {
    super(`LLM request failed: ${code}`);
    this.name = 'LlmError';
    this.code = code;
    this.retryable = retryable;
  }
}

export type SocketStatus = 'idle' | 'connecting' | 'ready' | 'reconnecting' | 'signed-out' | 'stopped';

export interface CloudSocketOptions {
  /** ws(s)://host/ws */
  url: string;
  auth: AuthSession;
  /** Defaults to the standard global WebSocket (React Native, Node ≥ 22, Electron). */
  createSocket?: (url: string) => WebSocket;
  backoffInitialMs?: number;
  backoffMaxMs?: number;
  /** Open + `auth` + `ready` must finish within this time. */
  connectTimeoutMs?: number;
  pingIntervalMs?: number;
  /** No frame at all for this long → the connection is considered dead and replaced. */
  deadAfterMs?: number;
  /** A request that got no event for this long is failed with `timeout` (it may be lost). */
  requestIdleTimeoutMs?: number;
}

interface Pending {
  sessionId: string;
  resolve: (done: DoneEvent) => void;
  reject: (error: Error) => void;
  lastEventAt: number;
}

type Listener<T> = (value: T) => void;

/**
 * The one WebSocket to the cloud (protocol.md). Keeps itself connected: authenticates with the
 * first message, reconnects with exponential backoff, resumes sessions with requests in flight
 * (the server replays their text or repeats `done`), pings to detect dead links, and queues
 * requests while offline. `request()` resolves with `done` — the agent's source of truth;
 * the UI listens to every event via `onEvent()` and draws deltas by channel.
 */
export class CloudSocket {
  private readonly options: Required<Omit<CloudSocketOptions, 'auth' | 'url' | 'createSocket'>>;
  private readonly url: string;
  private readonly auth: AuthSession;
  private readonly createSocket: (url: string) => WebSocket;

  private ws: WebSocket | undefined;
  private state: SocketStatus = 'idle';
  private attempt = 0;
  private forceRefresh = false;
  private lastFrameAt = 0;
  private readonly outbox: ClientMessage[] = [];
  private readonly pending = new Map<string, Pending>();
  private readonly eventListeners = new Set<Listener<ServerEvent>>();
  private readonly statusListeners = new Set<Listener<SocketStatus>>();
  private connectTimer: ReturnType<typeof setTimeout> | undefined;
  private retryTimer: ReturnType<typeof setTimeout> | undefined;
  private pingTimer: ReturnType<typeof setInterval> | undefined;

  constructor(options: CloudSocketOptions) {
    this.url = options.url;
    this.auth = options.auth;
    this.createSocket = options.createSocket ?? ((url) => new WebSocket(url));
    this.options = {
      backoffInitialMs: options.backoffInitialMs ?? 500,
      backoffMaxMs: options.backoffMaxMs ?? 15_000,
      connectTimeoutMs: options.connectTimeoutMs ?? 10_000,
      pingIntervalMs: options.pingIntervalMs ?? 10_000,
      deadAfterMs: options.deadAfterMs ?? 25_000,
      requestIdleTimeoutMs: options.requestIdleTimeoutMs ?? 90_000,
    };
  }

  get status(): SocketStatus {
    return this.state;
  }

  start(): void {
    if (this.state !== 'idle' && this.state !== 'signed-out' && this.state !== 'stopped') return;
    this.attempt = 0;
    void this.connect();
  }

  /** Closes for good: pending requests fail, no reconnects. */
  stop(): void {
    this.setStatus('stopped');
    this.clearTimers();
    this.ws?.close(1000, 'client stop');
    this.ws = undefined;
    this.failAll(new Error('Connection stopped'));
  }

  /** Drops the current connection and reconnects (e.g. the app came back to the foreground). */
  reconnectNow(): void {
    if (this.ws) this.ws.close(4000, 'client reconnect');
    else if (this.state === 'reconnecting') {
      clearTimeout(this.retryTimer);
      void this.connect();
    }
  }

  onEvent(listener: Listener<ServerEvent>): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  onStatus(listener: Listener<SocketStatus>): () => void {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  /** Sends an LLM request and resolves with its `done`; rejects with LlmError on `error`. */
  request(request: LlmRequest, signal?: AbortSignal): Promise<DoneEvent> {
    return new Promise<DoneEvent>((resolve, reject) => {
      if (this.state === 'stopped') return reject(new Error('Connection stopped'));
      if (this.state === 'signed-out') return reject(new AuthRequiredError());
      this.pending.set(request.requestId, {
        sessionId: request.sessionId,
        resolve,
        reject,
        lastEventAt: Date.now(),
      });
      // Cancel stops the generation between deltas; `done` with finishReason "cancelled" follows.
      signal?.addEventListener('abort', () => this.send({ v: 1, type: 'cancel', requestId: request.requestId }), {
        once: true,
      });
      this.send(request);
    });
  }

  private send(message: ClientMessage): void {
    if (this.state === 'ready' && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else if (message.type === 'llm.request' || message.type === 'cancel') {
      this.outbox.push(message);
    }
  }

  private async connect(): Promise<void> {
    this.setStatus(this.attempt === 0 ? 'connecting' : 'reconnecting');
    let token: string;
    try {
      token = await this.auth.accessToken({ force: this.forceRefresh });
    } catch (error) {
      if (error instanceof AuthRequiredError) {
        this.setStatus('signed-out');
        this.failAll(error);
        return;
      }
      this.scheduleReconnect(); // offline: the refresh call itself failed
      return;
    }
    if (this.state === 'stopped') return;

    const ws = this.createSocket(this.url);
    this.ws = ws;
    this.connectTimer = setTimeout(() => ws.close(4000, 'connect timeout'), this.options.connectTimeoutMs);
    ws.onopen = () => ws.send(JSON.stringify({ v: 1, type: 'auth', token }));
    ws.onmessage = (message) => this.onFrame(ws, String(message.data));
    ws.onerror = () => undefined; // a close event always follows
    ws.onclose = (event) => {
      if (this.ws !== ws) return;
      this.ws = undefined;
      this.clearTimers();
      if (this.state === 'stopped') return;
      if (event.code === CLOSE_UNAUTHORIZED) this.forceRefresh = true;
      this.scheduleReconnect();
    };
  }

  private onFrame(ws: WebSocket, raw: string): void {
    this.lastFrameAt = Date.now();
    const event = parseServerEvent(raw);
    if (!event) return;
    if (event.type === 'ready') {
      this.onReady(ws);
      return;
    }
    if ('requestId' in event && event.requestId) {
      const pending = this.pending.get(event.requestId);
      if (pending) {
        pending.lastEventAt = Date.now();
        if (event.type === 'done') {
          this.pending.delete(event.requestId);
          pending.resolve(event);
        } else if (event.type === 'error') {
          this.pending.delete(event.requestId);
          pending.reject(new LlmError(event.code, event.retryable));
        }
      } else if (event.type === 'done' || event.type === 'error') {
        // A repeat after `resume` of something already handled (protocol.md): not news.
        return;
      }
    }
    for (const listener of this.eventListeners) listener(event);
  }

  private onReady(ws: WebSocket): void {
    clearTimeout(this.connectTimer);
    this.attempt = 0;
    this.forceRefresh = false;
    this.setStatus('ready');
    const now = Date.now();
    // Sessions with requests in flight: the server replays text or repeats `done`.
    for (const sessionId of new Set([...this.pending.values()].map((p) => p.sessionId))) {
      ws.send(JSON.stringify({ v: 1, type: 'resume', sessionId }));
    }
    for (const pending of this.pending.values()) pending.lastEventAt = Math.max(pending.lastEventAt, now);
    for (const message of this.outbox.splice(0)) ws.send(JSON.stringify(message));
    this.pingTimer = setInterval(() => this.tick(ws), this.options.pingIntervalMs);
  }

  private tick(ws: WebSocket): void {
    const now = Date.now();
    if (now - this.lastFrameAt > this.options.deadAfterMs) {
      ws.close(4000, 'no frames'); // reconnect via onclose
      return;
    }
    ws.send(JSON.stringify({ v: 1, type: 'ping' }));
    for (const [requestId, pending] of this.pending) {
      if (now - pending.lastEventAt > this.options.requestIdleTimeoutMs) {
        this.pending.delete(requestId);
        pending.reject(new LlmError('timeout', true));
      }
    }
  }

  private scheduleReconnect(): void {
    const base = Math.min(this.options.backoffMaxMs, this.options.backoffInitialMs * 2 ** this.attempt);
    const delay = base * (0.5 + Math.random() * 0.5);
    this.attempt += 1;
    this.setStatus('reconnecting');
    this.retryTimer = setTimeout(() => void this.connect(), delay);
  }

  private clearTimers(): void {
    clearTimeout(this.connectTimer);
    clearTimeout(this.retryTimer);
    clearInterval(this.pingTimer);
  }

  private failAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    this.outbox.length = 0;
  }

  private setStatus(status: SocketStatus): void {
    if (this.state === status) return;
    this.state = status;
    for (const listener of this.statusListeners) listener(status);
  }
}
