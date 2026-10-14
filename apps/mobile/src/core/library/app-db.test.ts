import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { Library } from '@/core/library/library';
import { migrate } from '@/core/storage/migrate';
import { makeTestEnv, type TestEnv } from '@/platform/node/test-env';

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
