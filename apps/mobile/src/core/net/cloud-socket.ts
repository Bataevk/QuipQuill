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

/** The caller aborted the request (AbortSignal) while it could not be cancelled on the server. */
export class RequestAbortedError extends Error {
  constructor() {
    super('Request aborted');
    this.name = 'AbortError';
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
  /**
   * After `resume` the server answers at once for every request it knows (replay, thinking or
   * its `done`); one that gets nothing within this time is lost (its result outlived the
   * server's 15-minute buffer, or the server restarted) and fails with `timeout` at once.
   */
  resumeGraceMs?: number;
}

interface Pending {
  sessionId: string;
  /** Reached the server at least once (only then is there anything to resume or cancel). */
  sent: boolean;
  resolve: (done: DoneEvent) => void;
  reject: (error: Error) => void;
  lastEventAt: number;
  /** When it was resumed after a reconnect and has not heard from the server since. */
  resumedAt?: number;
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
      resumeGraceMs: options.resumeGraceMs ?? 15_000,
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

  /**
   * Drops the current connection and reconnects at once (the app came back to the foreground:
   * the OS may have killed the link while the app was suspended). Requests in flight resume.
   */
  reconnectNow(): void {
    if (this.ws) this.drop(this.ws, 'client reconnect', { immediately: true });
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

  /**
   * Sends an LLM request and resolves with its `done`; rejects with LlmError on `error`.
   * Abort while connected: the server stops between deltas and `done` (finishReason
   * "cancelled") follows with the text so far. Abort while offline: rejects at once with
   * RequestAbortedError; a request already sent gets its `cancel` after reconnecting.
   */
  request(request: LlmRequest, signal?: AbortSignal): Promise<DoneEvent> {
    return new Promise<DoneEvent>((resolve, reject) => {
      if (this.state === 'stopped') return reject(new Error('Connection stopped'));
      if (this.state === 'signed-out') return reject(new AuthRequiredError());
      if (signal?.aborted) return reject(new RequestAbortedError());
      const { requestId } = request;
      const onAbort = () => this.abort(requestId);
      const settle = () => signal?.removeEventListener('abort', onAbort);
      this.pending.set(requestId, {
        sessionId: request.sessionId,
        sent: false,
        resolve: (done) => (settle(), resolve(done)),
        reject: (error) => (settle(), reject(error)),
        lastEventAt: Date.now(),
      });
      signal?.addEventListener('abort', onAbort, { once: true });
      this.send(request);
    });
  }

  private abort(requestId: string): void {
    const pending = this.pending.get(requestId);
    if (!pending) return;
    if (this.state === 'ready') {
      this.send({ v: 1, type: 'cancel', requestId });
      return;
    }
    // Offline: nobody can confirm the cancel, so settle now and forget the request.
    this.pending.delete(requestId);
    const queued = this.outbox.findIndex((m) => m.type === 'llm.request' && m.requestId === requestId);
    if (queued >= 0) this.outbox.splice(queued, 1); // never sent: nothing to cancel
    else this.outbox.push({ v: 1, type: 'cancel', requestId }); // stop it after reconnecting
    pending.reject(new RequestAbortedError());
  }

  private send(message: ClientMessage): void {
    if (this.state === 'ready' && this.ws?.readyState === WebSocket.OPEN) {
      this.transmit(this.ws, message);
    } else if (message.type === 'llm.request' || message.type === 'cancel') {
      this.outbox.push(message);
    }
  }

  private transmit(ws: WebSocket, message: ClientMessage): void {
    ws.send(JSON.stringify(message));
    if (message.type === 'llm.request') {
      const pending = this.pending.get(message.requestId);
      if (pending) pending.sent = true;
    }
  }

  private async connect(): Promise<void> {
    // A first connection is "connecting"; restoring a lost one stays "reconnecting".
    this.setStatus(this.attempt === 0 && this.state !== 'reconnecting' ? 'connecting' : 'reconnecting');
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
    this.connectTimer = setTimeout(() => this.drop(ws, 'connect timeout'), this.options.connectTimeoutMs);
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
        pending.resumedAt = undefined;
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
    const inFlight = [...this.pending.values()].filter((p) => p.sent);
    for (const sessionId of new Set(inFlight.map((p) => p.sessionId))) {
      this.transmit(ws, { v: 1, type: 'resume', sessionId });
    }
    for (const pending of inFlight) pending.resumedAt = now;
    for (const pending of this.pending.values()) pending.lastEventAt = Math.max(pending.lastEventAt, now);
    for (const message of this.outbox.splice(0)) this.transmit(ws, message);
    this.pingTimer = setInterval(() => this.tick(ws), this.options.pingIntervalMs);
  }

  private tick(ws: WebSocket): void {
    const now = Date.now();
    if (now - this.lastFrameAt > this.options.deadAfterMs) {
      this.drop(ws, 'no frames');
      return;
    }
    this.transmit(ws, { v: 1, type: 'ping' });
    for (const [requestId, pending] of this.pending) {
      const lost = pending.resumedAt !== undefined && now - pending.resumedAt > this.options.resumeGraceMs;
      if (lost || now - pending.lastEventAt > this.options.requestIdleTimeoutMs) {
        this.pending.delete(requestId);
        pending.reject(new LlmError('timeout', true));
      }
    }
  }

  /**
   * Gives up on a connection the client found dead and reconnects, without waiting for its
   * close event: a socket that died while the app was suspended may never send one (iOS), and
   * waiting for it would leave every request hanging. Its late events are ignored.
   */
  private drop(ws: WebSocket, reason: string, { immediately = false } = {}): void {
    if (this.ws !== ws) return;
    this.ws = undefined;
    this.clearTimers();
    ws.onopen = null;
    ws.onmessage = null;
    ws.onclose = null;
    ws.onerror = null;
    try {
      ws.close(4000, reason);
    } catch {
      // already broken: nothing to close
    }
    if (this.state === 'stopped') return;
    if (immediately) {
      this.attempt = 0;
      this.setStatus('reconnecting');
      void this.connect();
    } else this.scheduleReconnect();
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
