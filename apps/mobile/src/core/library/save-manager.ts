import type { Db, DbDriver } from '../storage/db';
import { joinPath, type FileStore } from '../storage/files';
import { migrate } from '../storage/migrate';
import { newId, nowIso } from '../util/ids';
import type { AppRepository, SaveRecord } from './app-db';
import { PLAYER_ID } from '../world/model';
import { OUTDATED_SAVE_VERSION, SAVE_SCHEMA_STEPS } from './save-schema';

export const SAVES_DIR = 'saves';

/** The save cannot be opened: unknown id, or its file was deleted outside the app. */
export class SaveMissingError extends Error {
  readonly saveId: string;

  constructor(saveId: string, reason: string) {
    super(`Save ${saveId} cannot be opened: ${reason}`);
    this.name = 'SaveMissingError';
    this.saveId = saveId;
  }
}

/**
 * The save was played on an older world (a full copy of the story, ADR 4; the player as
 * `player:<id>`, ADR 43) and cannot be carried over. It can be started over or deleted.
 */
export class SaveOutdatedError extends Error {
  readonly saveId: string;

  constructor(saveId: string) {
    super(
      'This game was saved by an older version of QuipQuill and cannot be continued. Start it over or delete it.',
    );
    this.name = 'SaveOutdatedError';
    this.saveId = saveId;
  }
}

export interface NewSave {
  storyId: string;
  characterId: string | null;
  title: string;
}

/** A save is a copy of the story file plus the playthrough's tables: dynamic graph, journal, chats (ADR 42). */
export class SaveManager {
  private readonly driver: DbDriver;
  private readonly files: FileStore;
  private readonly library: AppRepository;

  constructor(driver: DbDriver, files: FileStore, library: AppRepository) {
    this.driver = driver;
    this.files = files;
    this.library = library;
  }

  async create(input: NewSave): Promise<SaveRecord> {
    const story = await this.library.getStory(input.storyId);
    if (!story) throw new Error(`Story ${input.storyId} is not in the library`);

    const id = newId();
    const relativePath = joinPath(SAVES_DIR, `${id}.db`);
    const path = this.absolute(relativePath);
    await this.files.ensureDir(this.absolute(SAVES_DIR));
    await this.files.copy(this.absolute(story.path), path);
    try {
      const db = await this.openFile(path);
      await db.close();
      const now = nowIso();
      const save: SaveRecord = {
        id,
        storyId: story.id,
        characterId: input.characterId,
        title: input.title,
        path: relativePath,
        createdAt: now,
        updatedAt: now,
      };
      await this.library.addSave(save);
      return save;
    } catch (error) {
      await this.files.remove(path);
      throw error;
    }
  }

  /**
   * Opens a save with its schema up to date. The caller closes the connection. Fails with
   * SaveMissingError if the file is gone: SQLite would silently create an empty database, and
   * with SaveOutdatedError if it was played on the old world schema.
   */
  async open(saveId: string): Promise<Db> {
    const save = await this.library.getSave(saveId);
    if (!save) throw new SaveMissingError(saveId, 'there is no such save');
    const path = this.absolute(save.path);
    if (!(await this.files.exists(path))) throw new SaveMissingError(saveId, `${save.path} is missing`);
    return this.openFile(path, saveId);
  }

  /**
   * "Start over": the save becomes a fresh copy of its story (the version in the library now),
   * keeping its id, title and character. The new copy is prepared beside the old one and moved
   * over it only when ready. Close the save's connection first.
   */
  async restart(saveId: string): Promise<SaveRecord> {
    const save = await this.library.getSave(saveId);
    if (!save) throw new SaveMissingError(saveId, 'there is no such save');
    const story = await this.library.getStory(save.storyId);
    if (!story) throw new Error(`Story ${save.storyId} is not in the library`);
    const path = this.absolute(save.path);
    const fresh = `${path}.restart`;
    await this.files.copy(this.absolute(story.path), fresh);
    try {
      const db = await this.openFile(fresh);
      await db.close();
      await this.files.move(fresh, path);
    } catch (error) {
      await this.files.remove(fresh);
      throw error;
    }
    const now = nowIso();
    await this.library.touchSave(saveId, now);
    return { ...save, updatedAt: now };
  }

  async delete(saveId: string): Promise<void> {
    const save = await this.library.getSave(saveId);
    if (!save) return;
    await this.library.deleteSave(saveId);
    await this.files.remove(this.absolute(save.path));
  }

  private async openFile(path: string, saveId?: string): Promise<Db> {
    const db = await this.driver.open(path);
    try {
      await db.exec('PRAGMA foreign_keys = ON');
      if (saveId && (await isOutdated(db))) throw new SaveOutdatedError(saveId);
      await migrate(db, SAVE_SCHEMA_STEPS);
      return db;
    } catch (error) {
      await db.close();
      throw error;
    }
  }

  private absolute(relativePath: string): string {
    return joinPath(this.files.dataDir, relativePath);
  }
}

/**
 * A save that has started on an older world: the state_* copies (schema v1), or a player that is
 * not the entity `player` (before ADR 43). Nothing to carry it over to.
 */
async function isOutdated(db: Db): Promise<boolean> {
  const version = Number((await db.get<{ user_version: number }>('PRAGMA user_version'))?.user_version ?? 0);
  if (version === OUTDATED_SAVE_VERSION) return !!(await db.get('SELECT 1 FROM events LIMIT 1'));
  if (version < 2) return false;
  return !!(await db.get("SELECT 1 FROM dyn_entities WHERE type = 'PLAYER' AND id <> ? LIMIT 1", [PLAYER_ID]));
}
