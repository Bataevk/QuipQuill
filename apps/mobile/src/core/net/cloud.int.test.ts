/**
 * Core ↔ local cloud (echo worker). Run with `npm run test:cloud` after `make infra-up`.
 * Stage 3.4 "done when": a request goes through, and a network drop mid-answer recovers via replay.
 */
import { HumanMessage } from '@langchain/core/messages';
import { createAgent } from 'langchain';
import { afterAll, afterEach, beforeAll, describe, expect, it } from 'vitest';

import { AuthSession, MemoryTokenStore } from '@/core/net/auth';
import { CloudSocket, LlmError, type SocketStatus } from '@/core/net/cloud-socket';
import { QueueChatModel } from '@/core/net/queue-chat-model';
import type { LlmRequest, ServerEvent } from '@/core/protocol/messages';
import { StreamText } from '@/core/protocol/stream-text';
import { newId } from '@/core/util/ids';
import { startTestCloud, TEST_INVITE, type TestCloud } from '@/platform/node/cloud-server';

const TEXT = Array.from({ length: 40 }, (_, i) => `word${i}`).join(' ');

let cloud: TestCloud;
let socket: CloudSocket | undefined;

beforeAll(async () => {
  cloud = await startTestCloud({ echoDelaySeconds: 0.03 });
}, 60_000);

afterAll(async () => {
  await cloud?.stop();
});

afterEach(() => {
  socket?.stop();
  socket = undefined;
});

async function signIn(): Promise<{ auth: AuthSession; store: MemoryTokenStore }> {
  const store = new MemoryTokenStore();
  const auth = new AuthSession(cloud.httpUrl, store);
  await auth.register(`ts-${newId()}@example.com`, 'long enough pass', TEST_INVITE);
  return { auth, store };
}

function connect(auth: AuthSession): { socket: CloudSocket; statuses: SocketStatus[]; events: ServerEvent[] } {
  const s = new CloudSocket({
    url: cloud.wsUrl,
    auth,
    backoffInitialMs: 50,
    pingIntervalMs: 300,
    deadAfterMs: 3000,
  });
  const statuses: SocketStatus[] = [];
  const events: ServerEvent[] = [];
  s.onStatus((status) => statuses.push(status));
  s.onEvent((event) => events.push(event));
  s.start();
  socket = s;
  return { socket: s, statuses, events };
}

function until(check: () => boolean, timeoutMs = 10_000): Promise<void> {
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
    }, 10);
  });
}

function rawRequest(text: string, requestId = newId()): LlmRequest {
  return {
    v: 1,
    type: 'llm.request',
    requestId,
    sessionId: 'save-1',
    channel: 'gm',
    purpose: 'gm',
    messages: [{ role: 'user', content: text }],
  };
}

describe('auth', () => {
  it('registers, logs in and rotates the refresh token', async () => {
    const { auth, store } = await signIn();
    const first = await store.load();
    const forced = await auth.accessToken({ force: true });
    const second = await store.load();
    expect(forced).toBe(second?.accessToken);
    expect(second?.refreshToken).not.toBe(first?.refreshToken);
  });
});

describe('QueueChatModel over the socket', () => {
  it('streams deltas on its channel and returns done.message', async () => {
    const { auth } = await signIn();
    const { socket: s, events } = connect(auth);
    const model = new QueueChatModel({ socket: s, sessionId: 'save-1', channel: 'npc:innkeeper', purpose: 'npc' });

    const answer = await model.invoke([new HumanMessage(TEXT)]);

    expect(answer.content).toBe(TEXT);
    expect(answer.response_metadata).toMatchObject({ finish_reason: 'stop' });
    const deltas = events.filter((e) => e.type === 'delta');
    expect(deltas.every((e) => e.channel === 'npc:innkeeper')).toBe(true);
    expect(deltas.map((e) => (e.type === 'delta' ? e.text : '')).join('')).toBe(TEXT);
  });

  it('runs inside createAgent (ADR 25)', async () => {
    const { auth } = await signIn();
    const { socket: s } = connect(auth);
    const agent = createAgent({
      model: new QueueChatModel({ socket: s, sessionId: 'save-1', channel: 'gm', purpose: 'gm' }),
      tools: [],
    });
    const result = await agent.invoke({ messages: [{ role: 'user', content: 'I look around.' }] });
    expect(result.messages.at(-1)?.content).toBe('I look around.');
  });

  it('recovers from a network drop mid-answer via replay', async () => {
    const { auth } = await signIn();
    const { socket: s, statuses, events } = connect(auth);
    const model = new QueueChatModel({ socket: s, sessionId: 'save-drop', channel: 'gm', purpose: 'gm' });

    const answer = model.invoke([new HumanMessage(TEXT)]);
    await until(() => events.filter((e) => e.type === 'delta').length >= 5);
    s.reconnectNow(); // the link dies; the worker keeps generating

    const message = await answer;
    expect(message.content).toBe(TEXT);
    expect(statuses).toEqual(expect.arrayContaining(['reconnecting', 'ready']));
    const replay = events.find((e) => e.type === 'replay');
    expect(replay).toBeDefined();

    const shown = new StreamText(replay && 'requestId' in replay ? replay.requestId : '');
    for (const event of events) shown.apply(event);
    expect(shown.text).toBe(TEXT);
    expect(shown.done).toBe(true);
  });
});

describe('CloudSocket', () => {
  it('refreshes the access token when the server answers 4401', async () => {
    const { auth, store } = await signIn();
    const tokens = (await store.load())!;
    await store.save({ ...tokens, accessToken: 'garbage' }); // looks valid locally
    const { socket: s, statuses } = connect(auth);

    await until(() => s.status === 'ready');
    expect(statuses).toContain('reconnecting');
    expect((await store.load())?.accessToken).not.toBe('garbage');
  });

  it('rejects with LlmError on a failed generation', async () => {
    const { auth } = await signIn();
    const { socket: s } = connect(auth);
    const failure = s.request(rawRequest('!fail'));
    await expect(failure).rejects.toBeInstanceOf(LlmError);
    await expect(failure).rejects.toMatchObject({ code: 'provider_error', retryable: true });
  });

  it('cancels a generation on abort', async () => {
    const { auth } = await signIn();
    const { socket: s, events } = connect(auth);
    const abort = new AbortController();
    const done = s.request(rawRequest(TEXT), abort.signal);
    await until(() => events.some((e) => e.type === 'delta'));
    abort.abort();
    const result = await done;
    expect(result.finishReason).toBe('cancelled');
    expect((result.message.content ?? '').split(' ').length).toBeLessThan(40);
  });
});
