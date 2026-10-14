/**
 * File system port. Paths are plain strings with `/` separators. Library records store paths
 * *relative* to `dataDir`: on iOS the app container path changes between installs and updates.
 */
export interface FileStore {
  /** Absolute directory for the app's own files (stories, saves, app.db). */
  readonly dataDir: string;
  exists(path: string): Promise<boolean>;
  /** Copies a file, replacing `to` if it exists. */
  copy(from: string, to: string): Promise<void>;
  /** Moves a file, replacing `to` if it exists. */
  move(from: string, to: string): Promise<void>;
  /** Deletes a file; no error if it is already gone. */
  remove(path: string): Promise<void>;
  /** Creates a directory with its parents; no error if it exists. */
  ensureDir(path: string): Promise<void>;
}

export function joinPath(...parts: string[]): string {
  return parts
    .filter((part) => part.length > 0)
    .map((part, i) => (i === 0 ? part.replace(/\/+$/, '') : part.replace(/^\/+|\/+$/g, '')))
    .join('/');
}
