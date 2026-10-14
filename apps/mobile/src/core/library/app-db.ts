import type { Db } from '../storage/db';
import { migrate } from '../storage/migrate';

/**
 * app.db: what lives outside any playthrough — characters, imported stories, saves.
 * Paths are relative to FileStore.dataDir (the iOS container path changes between installs).
 * Docs: docs/architecture/data-model.md#appdb-приложение
 */
export const APP_DB_FILE = 'app.db';

const STEPS = [
  `CREATE TABLE characters (
     id          TEXT PRIMARY KEY,
     name        TEXT NOT NULL,
     sex         TEXT,
     age         INTEGER,
     race        TEXT,
     description TEXT NOT NULL DEFAULT '',
     created_at  TEXT NOT NULL
   ) STRICT;
   CREATE TABLE stories (
     id            TEXT PRIMARY KEY,         -- meta.story_id: stable across rebuilds (ADR 24)
     title         TEXT NOT NULL,
     path          TEXT NOT NULL,            -- relative to dataDir
     built_at      TEXT NOT NULL DEFAULT '', -- meta.built_at of the imported version
     vector_index  TEXT NOT NULL CHECK (vector_index IN ('sqlite-vec', 'brute-force')),
     imported_at   TEXT NOT NULL
   ) STRICT;
   CREATE TABLE saves (
     id           TEXT PRIMARY KEY,
     story_id     TEXT NOT NULL REFERENCES stories (id),
     character_id TEXT REFERENCES characters (id),
     title        TEXT NOT NULL,
     path         TEXT NOT NULL,             -- relative to dataDir
     created_at   TEXT NOT NULL,
     updated_at   TEXT NOT NULL
   ) STRICT;
   CREATE INDEX saves_story ON saves (story_id);`,
];

export async function migrateAppDb(db: Db): Promise<void> {
  await db.exec('PRAGMA foreign_keys = ON');
  await migrate(db, STEPS);
}

export interface Character {
  id: string;
  name: string;
  sex: string | null;
  age: number | null;
  race: string | null;
  description: string;
  createdAt: string;
}

export interface StoryRecord {
  id: string;
  title: string;
  path: string;
  builtAt: string;
  vectorIndex: 'sqlite-vec' | 'brute-force';
  importedAt: string;
}

export interface SaveRecord {
  id: string;
  storyId: string;
  characterId: string | null;
  title: string;
  path: string;
  createdAt: string;
  updatedAt: string;
}

type StoryRow = {
  id: string;
  title: string;
  path: string;
  built_at: string;
  vector_index: StoryRecord['vectorIndex'];
  imported_at: string;
};
type SaveRow = {
  id: string;
  story_id: string;
  character_id: string | null;
  title: string;
  path: string;
  created_at: string;
  updated_at: string;
};
type CharacterRow = {
  id: string;
  name: string;
  sex: string | null;
  age: number | null;
  race: string | null;
  description: string;
  created_at: string;
};

/** Typed access to app.db. Holds an open connection owned by the caller. */
export class AppRepository {
  private readonly db: Db;

  constructor(db: Db) {
    this.db = db;
  }

  // --- characters

  async addCharacter(character: Character): Promise<void> {
    await this.db.run(
      `INSERT INTO characters (id, name, sex, age, race, description, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [
        character.id,
        character.name,
        character.sex,
        character.age,
        character.race,
        character.description,
        character.createdAt,
      ],
    );
  }

  async listCharacters(): Promise<Character[]> {
    const rows = await this.db.all<CharacterRow>('SELECT * FROM characters ORDER BY created_at');
    return rows.map((row) => ({
      id: row.id,
      name: row.name,
      sex: row.sex,
      age: row.age === null ? null : Number(row.age),
      race: row.race,
      description: row.description,
      createdAt: row.created_at,
    }));
  }

  // --- stories

  /** Insert, or replace the record of the same story_id (a newer build of the same story). */
  async upsertStory(story: StoryRecord): Promise<void> {
    await this.db.run(
      `INSERT INTO stories (id, title, path, built_at, vector_index, imported_at)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT (id) DO UPDATE SET title = excluded.title, path = excluded.path,
         built_at = excluded.built_at, vector_index = excluded.vector_index,
         imported_at = excluded.imported_at`,
      [story.id, story.title, story.path, story.builtAt, story.vectorIndex, story.importedAt],
    );
  }

  async getStory(id: string): Promise<StoryRecord | undefined> {
    const row = await this.db.get<StoryRow>('SELECT * FROM stories WHERE id = ?', [id]);
    return row && toStory(row);
  }

  async listStories(): Promise<StoryRecord[]> {
    const rows = await this.db.all<StoryRow>('SELECT * FROM stories ORDER BY title');
    return rows.map(toStory);
  }

  // --- saves

  async addSave(save: SaveRecord): Promise<void> {
    await this.db.run(
      `INSERT INTO saves (id, story_id, character_id, title, path, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [
        save.id,
        save.storyId,
        save.characterId,
        save.title,
        save.path,
        save.createdAt,
        save.updatedAt,
      ],
    );
  }

  async getSave(id: string): Promise<SaveRecord | undefined> {
    const row = await this.db.get<SaveRow>('SELECT * FROM saves WHERE id = ?', [id]);
    return row && toSave(row);
  }

  async listSaves(): Promise<SaveRecord[]> {
    const rows = await this.db.all<SaveRow>('SELECT * FROM saves ORDER BY updated_at DESC');
    return rows.map(toSave);
  }

  async touchSave(id: string, updatedAt: string): Promise<void> {
    await this.db.run('UPDATE saves SET updated_at = ? WHERE id = ?', [updatedAt, id]);
  }

  async deleteSave(id: string): Promise<void> {
    await this.db.run('DELETE FROM saves WHERE id = ?', [id]);
  }
}

function toStory(row: StoryRow): StoryRecord {
  return {
    id: row.id,
    title: row.title,
    path: row.path,
    builtAt: row.built_at,
    vectorIndex: row.vector_index,
    importedAt: row.imported_at,
  };
}

function toSave(row: SaveRow): SaveRecord {
  return {
    id: row.id,
    storyId: row.story_id,
    characterId: row.character_id,
    title: row.title,
    path: row.path,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}
