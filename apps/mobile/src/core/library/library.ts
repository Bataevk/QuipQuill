import { withDb, type Db, type DbDriver } from '../storage/db';
import { joinPath, type FileStore } from '../storage/files';
import { readStoryMeta, type StoryMeta } from '../story/story-meta';
import { APP_DB_FILE, AppRepository, migrateAppDb } from './app-db';
import { SaveManager } from './save-manager';
import { StoryImporter } from './story-importer';

/** Everything outside a playthrough, over one app.db connection. Built by the platform layer. */
export class Library {
  readonly repository: AppRepository;
  readonly importer: StoryImporter;
  readonly saves: SaveManager;
  private readonly db: Db;
  private readonly driver: DbDriver;
  private readonly files: FileStore;

  private constructor(db: Db, driver: DbDriver, files: FileStore) {
    this.db = db;
    this.driver = driver;
    this.files = files;
    this.repository = new AppRepository(db);
    this.importer = new StoryImporter(driver, files, this.repository);
    this.saves = new SaveManager(driver, files, this.repository);
  }

  static async open(driver: DbDriver, files: FileStore): Promise<Library> {
    await files.ensureDir(files.dataDir);
    const db = await driver.open(joinPath(files.dataDir, APP_DB_FILE));
    try {
      await migrateAppDb(db);
    } catch (error) {
      await db.close();
      throw error;
    }
    return new Library(db, driver, files);
  }

  /** The authored texts of an imported story (title, world summary, style), read from its file. */
  async storyMeta(storyId: string): Promise<StoryMeta | undefined> {
    const story = await this.repository.getStory(storyId);
    if (!story) return undefined;
    return withDb(this.driver, joinPath(this.files.dataDir, story.path), readStoryMeta, { readOnly: true });
  }

  async close(): Promise<void> {
    await this.db.close();
  }
}
