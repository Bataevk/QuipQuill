import { copyFile, stat, writeFile } from 'node:fs/promises';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AuthSession, MemoryTokenStore } from '@/core/net/auth';
import type { Downloader } from '@/core/net/download';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

import { StoryCatalog, StoryDownloadError, type CatalogStory } from './catalog';

let env: TestEnv;
let size: number;

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

/** A downloader that "downloads" the demo story (or `bytes`), reporting progress. */
function fakeDownloader(bytes?: string): Downloader & { calls: { url: string; auth?: string }[] } {
  const calls: { url: string; auth?: string }[] = [];
  return {
    calls,
    async download(url, destination, options) {
      calls.push({ url, auth: options.headers?.Authorization });
      if (bytes === undefined) await copyFile(DEMO_STORY, destination);
      else await writeFile(destination, bytes);
      options.onProgress?.({ received: size / 2, total: size });
      options.onProgress?.({ received: size, total: size });
    },
  };
}

async function catalogWith(downloader: Downloader) {
  const store = new MemoryTokenStore();
  await store.save({ accessToken: 'live', refreshToken: 'r', accessExpiresAt: Date.now() + 600_000 });
  const auth = new AuthSession('http://cloud', store);
  return new StoryCatalog({
    baseUrl: 'http://cloud/',
    auth,
    downloader,
    files: env.files,
    importer: env.library.importer,
    repository: env.library.repository,
  });
}

const entry = (over: Partial<CatalogStory> = {}): CatalogStory => ({
  id: DEMO_STORY_ID,
  version: 'abc',
  title: 'The Brass Vault',
  tagline: '',
  summary: '',
  heroes: [],
  builtAt: '2099-01-01T00:00:00+00:00',
  size,
  ...over,
});

beforeEach(async () => {
  env = await makeTestEnv();
  size = (await stat(DEMO_STORY)).size;
});

afterEach(async () => {
  vi.unstubAllGlobals();
  await env?.cleanup();
});

describe('story catalogue (ADR 61)', () => {
  it('lists the stories with their heroes, with the access token', async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
      json(200, [
        {
          id: DEMO_STORY_ID,
          version: 'v1',
          title: 'The Brass Vault',
          tagline: 'Steal it.',
          summary: 'A heist.',
          heroes: [{ key: 'ada', name: 'Ada', description: 'A locksmith.' }, { key: 'broken' }],
          built_at: '2026-10-21T00:00:00+00:00',
          size: 10,
        },
      ]),
    );
    vi.stubGlobal('fetch', fetchMock);
    const [story] = await (await catalogWith(fakeDownloader())).list();
    expect(story).toMatchObject({ id: DEMO_STORY_ID, tagline: 'Steal it.', builtAt: '2026-10-21T00:00:00+00:00', size: 10 });
    expect(story.heroes.map((h) => h.key)).toEqual(['ada']);
    expect(fetchMock.mock.calls[0][0]).toBe('http://cloud/stories');
    expect((fetchMock.mock.calls[0][1]?.headers as Record<string, string>).Authorization).toBe('Bearer live');
  });

  it('downloads and imports a story, reporting progress, and leaves no partial file', async () => {
    const downloader = fakeDownloader();
    const catalog = await catalogWith(downloader);
    const shares: number[] = [];
    const record = await catalog.install(entry(), { onProgress: (s) => shares.push(s) });

    expect(record.id).toBe(DEMO_STORY_ID);
    expect(await env.library.repository.getStory(DEMO_STORY_ID)).toBeDefined();
    expect(shares).toEqual([0.5, 1]);
    expect(downloader.calls).toEqual([{ url: `http://cloud/stories/${DEMO_STORY_ID}/file`, auth: 'Bearer live' }]);
    expect(await env.files.exists(`${env.files.dataDir}/downloads/${DEMO_STORY_ID}.qqstory.part`)).toBe(false);
  });

  it('refuses a download that stopped short, importing nothing', async () => {
    const catalog = await catalogWith(fakeDownloader('SQLite format 3 but cut'));
    await expect(catalog.install(entry())).rejects.toBeInstanceOf(StoryDownloadError);
    expect(await env.library.repository.listStories()).toEqual([]);
  });

  it('knows a story it has, one it lacks, and a newer build', async () => {
    expect(StoryCatalog.state(entry(), undefined)).toBe('available');
    const record = (await env.library.importer.import(DEMO_STORY)).story;
    expect(StoryCatalog.state(entry({ builtAt: record.builtAt }), record)).toBe('installed');
    expect(StoryCatalog.state(entry(), record)).toBe('update');
  });
});
