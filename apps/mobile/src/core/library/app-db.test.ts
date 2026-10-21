import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { CharacterInUseError } from '@/core/library/app-db';
import { Library } from '@/core/library/library';
import { migrate } from '@/core/storage/migrate';
import { DEMO_STORY, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

let env: TestEnv;

afterEach(async () => {
  await env?.cleanup();
});

describe('app.db', () => {
  it('stores characters and survives reopening (migrations run once)', async () => {
    env = await makeTestEnv();
    await env.library.repository.addCharacter({
      id: 'c1',
      name: 'Aria',
      sex: 'female',
      age: 27,
      race: 'elf',
      description: 'A wandering cartographer.',
      createdAt: '2026-10-15T00:00:00.000Z',
    });

    const reopened = await Library.open(env.driver, env.files);
    try {
      expect(await reopened.repository.listCharacters()).toEqual([
        expect.objectContaining({ id: 'c1', name: 'Aria', age: 27, race: 'elf' }),
      ]);
    } finally {
      await reopened.close();
    }
  });

  it('edits a hero, and deletes one only when no game plays it (stage D.7)', async () => {
    env = await makeTestEnv();
    const repo = env.library.repository;
    const hero = { id: 'h1', name: 'Wren', sex: null, age: null, race: 'half-elf', description: 'A locksmith.', createdAt: '2026-10-20T00:00:00.000Z' };
    const own = { ...hero, storyId: null };
    await repo.addCharacter(hero);
    await repo.updateCharacter({ ...hero, name: 'Wren Ashby', age: 31 });
    expect(await repo.getCharacter('h1')).toEqual({ ...own, name: 'Wren Ashby', age: 31 });

    const story = (await env.library.importer.import(DEMO_STORY)).story;
    const save = await env.library.saves.create({ storyId: story.id, characterId: 'h1', title: 'Vault — Wren' });
    await expect(repo.deleteCharacter('h1')).rejects.toBeInstanceOf(CharacterInUseError);
    expect(await repo.getCharacter('h1')).toBeDefined();

    await env.library.saves.delete(save.id);
    await repo.deleteCharacter('h1');
    expect(await repo.getCharacter('h1')).toBeUndefined();
  });

  it('keeps settings across reopening; a key is overwritten, not duplicated', async () => {
    env = await makeTestEnv();
    expect(await env.library.repository.getSetting('theme')).toBeUndefined();
    await env.library.repository.setSetting('theme', 'dark');
    await env.library.repository.setSetting('theme', 'light');

    const reopened = await Library.open(env.driver, env.files);
    try {
      expect(await reopened.repository.getSetting('theme')).toBe('light');
    } finally {
      await reopened.close();
    }
  });

  it('reads the meta of an imported story from its file', async () => {
    env = await makeTestEnv();
    const story = (await env.library.importer.import(DEMO_STORY)).story;
    const meta = await env.library.storyMeta(story.id);
    expect(meta?.title).toBe(story.title);
    expect(meta?.worldSummary).not.toBe('');
    expect(await env.library.storyMeta('no-such-story')).toBeUndefined();
  });

  it('refuses a database written by a newer app', async () => {
    env = await makeTestEnv();
    const db = await env.driver.open(join(env.dir, 'newer.db'));
    try {
      await db.exec('PRAGMA user_version = 5');
      await expect(migrate(db, ['SELECT 1'])).rejects.toThrow(/newer than this app/);
    } finally {
      await db.close();
    }
  });

  it('rolls back a failed migration step', async () => {
    env = await makeTestEnv();
    const db = await env.driver.open(join(env.dir, 'broken.db'));
    try {
      await expect(migrate(db, ['CREATE TABLE a (x); CREATE TABLE a (y);'])).rejects.toThrow();
      expect((await db.get<{ user_version: number }>('PRAGMA user_version'))?.user_version).toBe(0);
      expect(await db.get("SELECT name FROM sqlite_master WHERE name = 'a'")).toBeUndefined();
    } finally {
      await db.close();
    }
  });
});
