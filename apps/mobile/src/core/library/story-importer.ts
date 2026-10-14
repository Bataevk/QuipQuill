import { withDb, type DbDriver } from '../storage/db';
import { joinPath, type FileStore } from '../storage/files';
import { readStoryMeta, type StoryMeta } from '../story/story-meta';
import { nowIso } from '../util/ids';
import { SqliteVecIndex } from '../vector/sqlite-vec-index';
import type { AppRepository, StoryRecord } from './app-db';

export const STORIES_DIR = 'stories';

export interface ImportResult {
  story: StoryRecord;
  meta: StoryMeta;
  /** True if an earlier version of the same story (same story_id) was replaced. */
  replaced: boolean;
}

/**
 * Brings a .qqstory into the app: checks it, copies it to `stories/<story_id>.qqstory` and builds
 * the vector index in the copy (the original is never modified). A newer build of the same story
 * replaces the old file; existing saves are unaffected, each has its own copy.
 */
export class StoryImporter {
  private readonly driver: DbDriver;
  private readonly files: FileStore;
  private readonly library: AppRepository;

  constructor(driver: DbDriver, files: FileStore, library: AppRepository) {
    this.driver = driver;
    this.files = files;
    this.library = library;
  }

  /** `sourcePath` is an absolute path (e.g. what the system file picker returned). */
  async import(sourcePath: string): Promise<ImportResult> {
    // Throws StoryFormatError for anything that is not a playable story file.
    const meta = await withDb(this.driver, sourcePath, readStoryMeta, { readOnly: true });

    const relativePath = joinPath(STORIES_DIR, `${meta.storyId}.qqstory`);
    const target = joinPath(this.files.dataDir, relativePath);
    const staging = `${target}.importing`;
    await this.files.ensureDir(joinPath(this.files.dataDir, STORIES_DIR));
    try {
      await this.files.copy(sourcePath, staging);
      const hasVec = await withDb(this.driver, staging, (db) =>
        SqliteVecIndex.build(db, meta.embedDim),
      );
      await this.files.move(staging, target);

      const replaced = (await this.library.getStory(meta.storyId)) !== undefined;
      const story: StoryRecord = {
        id: meta.storyId,
        title: meta.title,
        path: relativePath,
        builtAt: meta.builtAt,
        vectorIndex: hasVec ? 'sqlite-vec' : 'brute-force',
        importedAt: nowIso(),
      };
      await this.library.upsertStory(story);
      return { story, meta, replaced };
    } finally {
      await this.files.remove(staging);
    }
  }
}
