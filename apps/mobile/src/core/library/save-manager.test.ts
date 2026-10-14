import { rm } from 'node:fs/promises';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { SaveMissingError } from '@/core/library/save-manager';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

let env: TestEnv;

afterEach(async () => {
  await env?.cleanup();
});

describe('SaveManager', () => {
  it('creates a save as a copy of the story with the state tables added', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const save = await env.library.saves.create({
      storyId: DEMO_STORY_ID,
      characterId: null,
      title: 'First run',
    });

    expect(save.path).toBe(`saves/${save.id}.db`);
    expect(await env.files.exists(join(env.files.dataDir, save.path))).toBe(true);
    expect(await env.library.repository.listSaves()).toEqual([save]);

    const db = await env.library.saves.open(save.id);
    try {
      const tables = (
        await db.all<{ name: string }>("SELECT name FROM sqlite_master WHERE type = 'table'")
      ).map((row) => row.name);
      expect(tables).toEqual(
        expect.arrayContaining([
          'entities',
          'relations',
          'embeddings',
          'meta',
          'state_entities',
          'state_relations',
          'events',
          'notes',
          'model_messages',
          'transcript',
        ]),
      );
      expect((await db.get<{ n: number }>('SELECT count(*) AS n FROM entities'))?.n).toBe(74);
      expect((await db.get<{ user_version: number }>('PRAGMA user_version'))?.user_version).toBe(1);
    } finally {
      await db.close();
    }
  });

  it('reopens a save without re-running its schema', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 's' });
    for (let i = 0; i < 2; i++) await (await env.library.saves.open(save.id)).close();
  });

  it('keeps a save playable after its story is re-imported', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 's' });
    await env.library.importer.import(DEMO_STORY);
    await (await env.library.saves.open(save.id)).close();
  });

  it('refuses to open a save whose file is gone instead of creating an empty one', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 's' });
    const path = join(env.files.dataDir, save.path);
    await rm(path);

    await expect(env.library.saves.open(save.id)).rejects.toBeInstanceOf(SaveMissingError);
    expect(await env.files.exists(path)).toBe(false);
    await expect(env.library.saves.open('nope')).rejects.toThrow(/no such save/);
  });

  it('deletes the file and the record', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 's' });

    await env.library.saves.delete(save.id);
    expect(await env.library.repository.listSaves()).toEqual([]);
    expect(await env.files.exists(join(env.files.dataDir, save.path))).toBe(false);
  });

  it('refuses a story that is not in the library', async () => {
    env = await makeTestEnv();
    await expect(
      env.library.saves.create({ storyId: 'nope', characterId: null, title: 's' }),
    ).rejects.toThrow(/not in the library/);
  });
});
