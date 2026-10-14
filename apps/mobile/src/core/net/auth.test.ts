import { afterEach, describe, expect, it, vi } from 'vitest';

import { AuthRequiredError, AuthSession, MemoryTokenStore } from '@/core/net/auth';
import { CloudEmbeddings } from '@/core/net/embeddings';

afterEach(() => {
  vi.unstubAllGlobals();
});

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

const pair = (n: number) => ({
  access_token: `access-${n}`,
  refresh_token: `refresh-${n}`,
  token_type: 'bearer',
  expires_in: 900,
});

async function expiredSession() {
  const store = new MemoryTokenStore();
  await store.save({ accessToken: 'old', refreshToken: 'refresh-0', accessExpiresAt: 0 });
  return { store, auth: new AuthSession('http://cloud', store) };
}

describe('AuthSession', () => {
  it('shares one refresh between concurrent callers (the refresh token rotates)', async () => {
    const fetchMock = vi.fn(async () => json(200, pair(1)));
    vi.stubGlobal('fetch', fetchMock);
    const { auth, store } = await expiredSession();

    const tokens = await Promise.all([auth.accessToken(), auth.accessToken(), auth.accessToken()]);

    expect(tokens).toEqual(['access-1', 'access-1', 'access-1']);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect((await store.load())?.refreshToken).toBe('refresh-1');
  });

  it('signs out when the refresh token is rejected', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json(401, { detail: 'refresh_token_reused' })));
    const { auth, store } = await expiredSession();

    await expect(auth.accessToken()).rejects.toBeInstanceOf(AuthRequiredError);
    expect(await store.load()).toBeNull();
  });

  it('keeps the tokens when the network fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('Network request failed'))));
    const { auth, store } = await expiredSession();

    await expect(auth.accessToken()).rejects.toThrow('Network request failed');
    expect((await store.load())?.refreshToken).toBe('refresh-0');
  });
});

describe('CloudEmbeddings', () => {
  it('splits big inputs into batches of 64 and keeps the order', async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const { input } = JSON.parse(String(init.body)) as { input: string[] };
      return json(200, { model: 'm', vectors: input.map((text) => [Number(text), 0.5]) });
    });
    vi.stubGlobal('fetch', fetchMock);
    const store = new MemoryTokenStore();
    await store.save({ accessToken: 'a', refreshToken: 'r', accessExpiresAt: Date.now() + 3_600_000 });
    const embeddings = new CloudEmbeddings('http://cloud', new AuthSession('http://cloud', store), 'm');

    const vectors = await embeddings.embed(Array.from({ length: 70 }, (_, i) => String(i)));

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(vectors).toHaveLength(70);
    expect(vectors[69]).toBeInstanceOf(Float32Array);
    expect(Array.from(vectors[69])).toEqual([69, 0.5]);
    const [, init] = fetchMock.mock.calls[0];
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer a');
  });

  it('refreshes once and retries when the access token is rejected', async () => {
    const answers = [
      json(401, { detail: 'not_authenticated' }),
      json(200, pair(2)),
      json(200, { model: 'm', vectors: [[1, 2]] }),
    ];
    const fetchMock = vi.fn(async (_url: string) => answers.shift()!);
    vi.stubGlobal('fetch', fetchMock);
    const store = new MemoryTokenStore();
    await store.save({ accessToken: 'stale', refreshToken: 'r', accessExpiresAt: Date.now() + 3_600_000 });
    const embeddings = new CloudEmbeddings('http://cloud', new AuthSession('http://cloud', store), 'm');

    expect((await embeddings.embed(['x']))[0]).toEqual(Float32Array.from([1, 2]));
    expect(fetchMock.mock.calls.map(([url]) => String(url))).toEqual([
      'http://cloud/embeddings',
      'http://cloud/auth/refresh',
      'http://cloud/embeddings',
    ]);
  });
});
