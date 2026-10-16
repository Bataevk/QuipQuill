import { rm } from 'node:fs/promises';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { SaveMissingError, SaveOutdatedError } from '@/core/library/save-manager';
import { SAVE_SCHEMA_STEPS } from '@/core/library/save-schema';
import { migrate } from '@/core/storage/migrate';
import { DEMO_STORY, DEMO_STORY_ID, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

let env: TestEnv;

afterEach(async () => {
  await env?.cleanup();
});

describe('SaveManager', () => {
  it('creates a save as a copy of the story with the dynamic graph added', async () => {
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
          'dyn_entities',
          'dyn_relations',
          'events',
          'notes',
          'model_messages',
          'transcript',
        ]),
      );
      expect((await db.get<{ n: number }>('SELECT count(*) AS n FROM entities'))?.n).toBe(74);
      const version = await db.get<{ user_version: number }>('PRAGMA user_version');
      expect(version?.user_version).toBe(SAVE_SCHEMA_STEPS.length);
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

  it('starts a save over from its story, keeping id, title and character', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 'Run' });
    let db = await env.library.saves.open(save.id);
    await db.run("INSERT INTO transcript (speaker, text, ts) VALUES ('player', 'hi', '2026-10-15')");
    await db.close();

    const restarted = await env.library.saves.restart(save.id);
    expect(restarted).toMatchObject({ id: save.id, title: 'Run', path: save.path, characterId: null });
    expect(restarted.updatedAt >= save.updatedAt).toBe(true);
    db = await env.library.saves.open(save.id);
    try {
      expect(await db.get('SELECT count(*) AS n FROM transcript')).toEqual({ n: 0 });
      expect(Number((await db.get<{ n: number }>('SELECT count(*) AS n FROM entities'))?.n)).toBeGreaterThan(0);
    } finally {
      await db.close();
    }
    expect(await env.files.exists(join(env.files.dataDir, `${save.path}.restart`))).toBe(false);
    expect(await env.library.repository.listSaves()).toHaveLength(1);
  });

  it('refuses a game played on the old schema, and lets it start over (ADR 42)', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 'Old' });
    await writeV1Save(join(env.files.dataDir, save.path), { started: true });

    await expect(env.library.saves.open(save.id)).rejects.toBeInstanceOf(SaveOutdatedError);
    await env.library.saves.restart(save.id);
    const db = await env.library.saves.open(save.id);
    try {
      const version = await db.get<{ user_version: number }>('PRAGMA user_version');
      expect(version?.user_version).toBe(SAVE_SCHEMA_STEPS.length);
    } finally {
      await db.close();
    }
  });

  it('refuses a game whose player is not the entity "player" (before ADR 43)', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 'Old' });
    const db = await env.library.saves.open(save.id);
    await db.run(
      "INSERT INTO dyn_entities (id, type, name, description, props, updated_seq) VALUES ('player:c1', 'PLAYER', 'Aria', '', '{}', 1)",
    );
    await db.close();

    await expect(env.library.saves.open(save.id)).rejects.toBeInstanceOf(SaveOutdatedError);
    await env.library.saves.restart(save.id);
    await (await env.library.saves.open(save.id)).close();
  });

  it('carries over an old-schema save that never started', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 'New' });
    await writeV1Save(join(env.files.dataDir, save.path), { started: false });

    const db = await env.library.saves.open(save.id);
    try {
      const tables = (await db.all<{ name: string }>('SELECT name FROM sqlite_master')).map((row) => row.name);
      expect(tables).toEqual(expect.arrayContaining(['dyn_entities', 'world_entities', 'world_relations']));
      expect(tables).not.toContain('state_entities');
    } finally {
      await db.close();
    }
  });

  it('carries a v3 game over to actors: npc.reacted becomes entity.reacted, NPC chats go (ADR 43)', async () => {
    env = await makeTestEnv();
    await env.library.importer.import(DEMO_STORY);
    const save = await env.library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 'v3' });
    const path = join(env.files.dataDir, save.path);
    await env.files.copy(DEMO_STORY, path);
    const old = await env.driver.open(path);
    try {
      await migrate(old, SAVE_SCHEMA_STEPS.slice(0, 3));
      await old.run(
        `INSERT INTO events (turn, actor, kind, payload, ts) VALUES (1, 'gm', 'npc.reacted', ?, 'now')`,
        [JSON.stringify({ npc: 'npc:mags-holloway', situation: 'Hi.', reaction: '"Hm."' })],
      );
      await old.run("INSERT INTO notes (id, text, created_seq) VALUES (1, 'Meet Mags.', 1)");
      await old.run(
        `INSERT INTO model_messages (chat_id, seq, role, content) VALUES ('npc:mags-holloway', 0, 'user', '"Hi."')`,
      );
    } finally {
      await old.close();
    }

    const db = await env.library.saves.open(save.id);
    try {
      const event = await db.get<{ kind: string; payload: string }>('SELECT kind, payload FROM events');
      expect(event?.kind).toBe('entity.reacted');
      expect(JSON.parse(event!.payload)).toEqual({ entity: 'npc:mags-holloway', situation: 'Hi.', reaction: '"Hm."' });
      expect(await db.get('SELECT entity_id FROM notes WHERE id = 1')).toEqual({ entity_id: null });
      expect(await db.all("SELECT * FROM model_messages WHERE chat_id LIKE 'npc:%'")).toEqual([]);
    } finally {
      await db.close();
    }
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

/** Replaces a save file with a fresh story copy on schema v1 (state_* copies), started or not. */
async function writeV1Save(path: string, { started }: { started: boolean }): Promise<void> {
  await env.files.copy(DEMO_STORY, path);
  const db = await env.driver.open(path);
  try {
    await migrate(db, SAVE_SCHEMA_STEPS.slice(0, 1));
    if (started) {
      await db.run(
        "INSERT INTO events (turn, actor, kind, payload, ts) VALUES (0, 'system', 'session.started', '{}', 'now')",
      );
    }
  } finally {
    await db.close();
  }
}
