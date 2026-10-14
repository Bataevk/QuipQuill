/** CloudSocket against a scripted fake WebSocket: timing and offline paths without a server. */
import { afterEach, describe, expect, it } from 'vitest';

import { AuthRequiredError, AuthSession, MemoryTokenStore } from '@/core/net/auth';
import { CloudSocket, LlmError, RequestAbortedError } from '@/core/net/cloud-socket';
import type { ClientMessage, LlmRequest } from '@/core/protocol/messages';

/** A WebSocket the test drives: it opens, talks and closes only when told to. */
class FakeSocket {
  static readonly OPEN = 1;
  readyState = 0;
  readonly sent: ClientMessage[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;
  /** Answer `ping` with `heartbeat`, like the server does. */
  answerPings = true;

  open(): void {
    this.readyState = 1;
    this.onopen?.();
  }

  /** Open, then accept the `auth` the client sends first. */
  openAndReady(): void {
    this.open();
    this.push({ v: 1, type: 'ready' });
  }

  push(event: object): void {
    this.onmessage?.({ data: JSON.stringify(event) });
  }

  send(data: string): void {
    const message = JSON.parse(data) as ClientMessage;
    this.sent.push(message);
    if (message.type === 'ping' && this.answerPings) this.push({ v: 1, type: 'heartbeat' });
  }

  close(code = 1006): void {
    if (this.readyState === 3) return;
    this.readyState = 3;
    this.onclose?.({ code });
  }

  types(): string[] {
    return this.sent.map((m) => m.type);
  }
}

const TOKENS = { accessToken: 'a', refreshToken: 'r', accessExpiresAt: Date.now() + 3_600_000 };

let socket: CloudSocket | undefined;
afterEach(() => socket?.stop());

async function setup(options: { signedIn?: boolean } & Record<string, number | boolean> = {}) {
  const store = new MemoryTokenStore();
  if (options.signedIn !== false) await store.save(TOKENS);
  const sockets: FakeSocket[] = [];
  socket = new CloudSocket({
    url: 'ws://test/ws',
    auth: new AuthSession('http://test', store),
    createSocket: () => {
      const fake = new FakeSocket();
      sockets.push(fake);
      return fake as unknown as WebSocket;
    },
    backoffInitialMs: 5,
    pingIntervalMs: 20,
    deadAfterMs: 1000,
    requestIdleTimeoutMs: 1000,
    ...options,
  });
  socket.start();
  await until(() => sockets.length > 0 || socket!.status === 'signed-out');
  return { socket, sockets, last: () => sockets[sockets.length - 1] };
}

function until(check: () => boolean, timeoutMs = 2000): Promise<void> {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const timer = setInterval(() => {
      if (check()) {
        clearInterval(timer);
        resolve();
      } else if (Date.now() > deadline) {
        clearInterval(timer);
        reject(new Error('condition not met in time'));
      }
    }, 2);
  });
}

function llm(requestId: string): LlmRequest {
  return {
    v: 1,
    type: 'llm.request',
    requestId,
    sessionId: 'save-1',
    channel: 'gm',
    purpose: 'gm',
    messages: [{ role: 'user', content: 'hi' }],
  };
}

describe('CloudSocket offline', () => {
  it('authenticates first, then flushes what was queued', async () => {
    const { socket: s, last } = await setup();
    void s.request(llm('r1')).catch(() => undefined);
    expect(last().sent).toEqual([]); // not open yet: queued
    last().openAndReady();
    expect(last().types().slice(0, 2)).toEqual(['auth', 'llm.request']);
    expect(s.status).toBe('ready');
  });

  it('rejects an abort at once when the request never left the device', async () => {
    const { socket: s, last } = await setup();
    const abort = new AbortController();
    const pending = s.request(llm('r1'), abort.signal);
    abort.abort();
    await expect(pending).rejects.toBeInstanceOf(RequestAbortedError);

    last().openAndReady();
    expect(last().types()).toEqual(['auth']); // nothing to send, nothing to cancel
  });

  it('cancels an already sent request after reconnecting', async () => {
    const { socket: s, sockets, last } = await setup();
    last().openAndReady();
    const abort = new AbortController();
    const pending = s.request(llm('r1'), abort.signal);
    last().close(1006); // the network drops
    abort.abort();
    await expect(pending).rejects.toBeInstanceOf(RequestAbortedError);

    await until(() => sockets.length === 2);
    last().openAndReady();
    expect(last().types()).toEqual(['auth', 'cancel']); // no resume: the request is forgotten
  });
});

describe('CloudSocket liveness', () => {
  it('replaces a connection that went silent', async () => {
    const { socket: s, sockets, last } = await setup({ deadAfterMs: 60 });
    last().answerPings = false;
    last().openAndReady();
    await until(() => sockets.length === 2);
    expect(['connecting', 'reconnecting']).toContain(s.status);
    last().openAndReady();
    expect(s.status).toBe('ready');
  });

  it('fails a request that gets no events for too long', async () => {
    const { socket: s, last } = await setup({ requestIdleTimeoutMs: 80 });
    last().openAndReady();
    await expect(s.request(llm('r1'))).rejects.toMatchObject({ code: 'timeout', retryable: true });
    await expect(s.request(llm('r2'))).rejects.toBeInstanceOf(LlmError);
  });

  it('goes signed-out without tokens and fails requests', async () => {
    const { socket: s } = await setup({ signedIn: false });
    expect(s.status).toBe('signed-out');
    await expect(s.request(llm('r1'))).rejects.toBeInstanceOf(AuthRequiredError);
  });
});
