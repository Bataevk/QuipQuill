import { copyFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { withDb } from '@/core/storage/db';
import { StoryFormatError } from '@/core/story/story-meta';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

let env: TestEnv;

afterEach(async () => {
  await env?.cleanup();
});

async function tableNames(env: TestEnv, path: string): Promise<string[]> {
  return withDb(
    env.driver,
    path,
    async (db) =>
      (await db.all<{ name: string }>("SELECT name FROM sqlite_master WHERE type = 'table'")).map(
        (row) => row.name,
      ),
    { readOnly: true },
  );
}

describe('StoryImporter', () => {
  it('imports the demo story into the library and builds the vec0 index in its copy', async () => {
    env = await makeTestEnv();
    const { story, meta, replaced } = await env.library.importer.import(DEMO_STORY);

    expect(replaced).toBe(false);
    expect(story).toMatchObject({
      id: DEMO_STORY_ID,
      title: 'The Brass Vault',
      path: `stories/${DEMO_STORY_ID}.qqstory`,
      vectorIndex: 'sqlite-vec',
    });
    expect(meta.startLocation).toBe('loc:rusty-gear-tavern');
    expect(meta.embedDim).toBe(1536);
    expect(await env.library.repository.listStories()).toEqual([story]);

    const copy = join(env.files.dataDir, story.path);
    expect(await tableNames(env, copy)).toContain('vec_entities');
    expect(await tableNames(env, DEMO_STORY)).not.toContain('vec_entities'); // original untouched
    expect(await env.files.exists(`${copy}.importing`)).toBe(false);
  });

  it('falls back to brute force where sqlite-vec is not available', async () => {
    env = await makeTestEnv({ vec: false });
    const { story } = await env.library.importer.import(DEMO_STORY);
    expect(story.vectorIndex).toBe('brute-force');
  });

  it('replaces an earlier version of the same story instead of adding a second one', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const again = await env.library.importer.import(DEMO_STORY);

    expect(again.replaced).toBe(true);
    expect(await env.library.repository.listStories()).toHaveLength(1);
  });

  it('imports a story shipped with the app only when it is new or another build', async () => {
    env = await makeTestEnv();
    expect((await env.library.importer.importIfChanged(DEMO_STORY))?.replaced).toBe(false);
    expect(await env.library.importer.importIfChanged(DEMO_STORY)).toBeUndefined();

    const rebuilt = join(env.dir, 'rebuilt.qqstory');
    await copyFile(DEMO_STORY, rebuilt);
    await withDb(env.driver, rebuilt, (db) => db.run("UPDATE meta SET value = '2099-01-01T00:00:00Z' WHERE key = 'built_at'"));
    const updated = await env.library.importer.importIfChanged(rebuilt);
    expect(updated?.replaced).toBe(true);
    expect((await env.library.repository.getStory(DEMO_STORY_ID))?.builtAt).toBe('2099-01-01T00:00:00Z');
  });

  it('rejects files that are not stories, leaving nothing behind', async () => {
    env = await makeTestEnv();
    const text = join(env.dir, 'notes.qqstory');
    await writeFile(text, 'just some text, not SQLite');

    await expect(env.library.importer.import(text)).rejects.toBeInstanceOf(StoryFormatError);
    expect(await env.library.repository.listStories()).toEqual([]);
  });

  it('rejects an unsupported schema version', async () => {
    env = await makeTestEnv();
    const future = join(env.dir, 'future.qqstory');
    await copyFile(DEMO_STORY, future);
    await withDb(env.driver, future, (db) =>
      db.run("UPDATE meta SET value = '99' WHERE key = 'schema_version'"),
    );

    await expect(env.library.importer.import(future)).rejects.toThrow(/v99 is not supported/);
  });
});
