/**
 * The story catalogue of the cloud (ADR 61): what can be played, and getting it onto the device.
 * A downloaded file is imported as before (StoryImporter); the world still lives on the device.
 */
import type { AuthSession } from '../net/auth';
import type { Downloader, DownloadProgress } from '../net/download';
import { HttpError, requestJson } from '../net/http';
import { joinPath, type FileStore } from '../storage/files';
import { parseHeroes, type StoryHero } from '../story/story-meta';
import type { AppRepository, StoryRecord } from './app-db';
import type { StoryImporter } from './story-importer';

/** A story file is ~1–10 MB; on a slow phone network that still fits in two minutes. */
export const DOWNLOAD_TIMEOUT_MS = 120_000;
export const DOWNLOADS_DIR = 'downloads';

export interface CatalogStory {
  id: string;
  /** sha256 of the file in the cloud. */
  version: string;
  title: string;
  tagline: string;
  summary: string;
  heroes: StoryHero[];
  /** meta.built_at: compared with the imported copy to see a new build (ADR 24). */
  builtAt: string;
  /** Bytes. */
  size: number;
}

/** On the device: not yet, the same build, or an older build than the cloud's. */
export type CatalogState = 'available' | 'installed' | 'update';

/** The download stopped short or brought something else; `message` is safe to show. */
export class StoryDownloadError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = 'StoryDownloadError';
  }
}

type StoryJson = {
  id: string;
  version: string;
  title: string;
  tagline?: string;
  summary?: string;
  heroes?: unknown;
  built_at?: string;
  size: number;
};

export class StoryCatalog {
  private readonly baseUrl: string;
  private readonly auth: AuthSession;
  private readonly downloader: Downloader;
  private readonly files: FileStore;
  private readonly importer: StoryImporter;
  private readonly repository: AppRepository;

  constructor(deps: {
    baseUrl: string;
    auth: AuthSession;
    downloader: Downloader;
    files: FileStore;
    importer: StoryImporter;
    repository: AppRepository;
  }) {
    this.baseUrl = deps.baseUrl.replace(/\/+$/, '');
    this.auth = deps.auth;
    this.downloader = deps.downloader;
    this.files = deps.files;
    this.importer = deps.importer;
    this.repository = deps.repository;
  }

  async list(): Promise<CatalogStory[]> {
    const rows = await this.authorized((token) => requestJson<StoryJson[]>(`${this.baseUrl}/stories`, { token }));
    return rows.map((row) => ({
      id: row.id,
      version: row.version,
      title: row.title,
      tagline: row.tagline ?? '',
      summary: row.summary ?? '',
      heroes: parseHeroes(JSON.stringify(row.heroes ?? [])),
      builtAt: row.built_at ?? '',
      size: row.size,
    }));
  }

  static state(story: Pick<CatalogStory, 'builtAt'>, local: Pick<StoryRecord, 'builtAt'> | undefined): CatalogState {
    if (!local) return 'available';
    return story.builtAt && story.builtAt !== local.builtAt ? 'update' : 'installed';
  }

  /** Downloads and imports the story (a new build replaces the old copy; saves keep theirs). */
  async install(
    story: CatalogStory,
    options: { onProgress?: (share: number) => void; signal?: AbortSignal } = {},
  ): Promise<StoryRecord> {
    const dir = joinPath(this.files.dataDir, DOWNLOADS_DIR);
    const part = joinPath(dir, `${story.id}.qqstory.part`);
    await this.files.ensureDir(dir);
    const signal = options.signal
      ? AbortSignal.any([options.signal, AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS)])
      : AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS);
    const onProgress = (p: DownloadProgress) =>
      options.onProgress?.(Math.min(1, p.received / (p.total && p.total > 0 ? p.total : story.size || 1)));
    try {
      await this.authorized(async (token) => {
        await this.downloader.download(`${this.baseUrl}/stories/${encodeURIComponent(story.id)}/file`, part, {
          headers: { Authorization: `Bearer ${token}` },
          onProgress,
          signal,
        });
        const size = await this.files.size(part);
        if (size === 0 || size === null) throw new HttpError(401, 'empty_download'); // a refused token
        if (size !== story.size) {
          throw new StoryDownloadError('The story did not download completely. Check the connection and try again.');
        }
      });
      const { story: record } = await this.importer.import(part);
      return record;
    } catch (error) {
      if (error instanceof StoryDownloadError) throw error;
      if (signal.aborted && !options.signal?.aborted) {
        throw new StoryDownloadError('The download took too long. Check the connection and try again.', { cause: error });
      }
      throw error;
    } finally {
      await this.files.remove(part);
    }
  }

  /** Runs a cloud call with the access token; on 401 refreshes it once and tries again. */
  private async authorized<T>(call: (token: string) => Promise<T>): Promise<T> {
    try {
      return await call(await this.auth.accessToken());
    } catch (error) {
      if (!(error instanceof HttpError) || error.status !== 401) throw error;
      return call(await this.auth.accessToken({ force: true }));
    }
  }
}
