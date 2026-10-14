import type { Db, DbDriver } from '../storage/db';
import { joinPath, type FileStore } from '../storage/files';
import { migrate } from '../storage/migrate';
import { newId, nowIso } from '../util/ids';
import type { AppRepository, SaveRecord } from './app-db';
import { SAVE_SCHEMA_STEPS } from './save-schema';

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

export interface NewSave {
  storyId: string;
  characterId: string | null;
  title: string;
}

/** A save is a copy of the story file plus the game's state tables (ADR 4). */
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
   * SaveMissingError if the file is gone: SQLite would silently create an empty database.
   */
  async open(saveId: string): Promise<Db> {
    const save = await this.library.getSave(saveId);
    if (!save) throw new SaveMissingError(saveId, 'there is no such save');
    const path = this.absolute(save.path);
    if (!(await this.files.exists(path))) throw new SaveMissingError(saveId, `${save.path} is missing`);
    return this.openFile(path);
  }

  async delete(saveId: string): Promise<void> {
    const save = await this.library.getSave(saveId);
    if (!save) return;
    await this.library.deleteSave(saveId);
    await this.files.remove(this.absolute(save.path));
  }

  private async openFile(path: string): Promise<Db> {
    const db = await this.driver.open(path);
    try {
      await db.exec('PRAGMA foreign_keys = ON');
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
