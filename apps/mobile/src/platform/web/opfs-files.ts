/**
 * FileStore port on the browser's storage pool (stage W.2, ADR 69). Files are pool entries named
 * by absolute path under `/qq`; the pool has no directories, so `ensureDir` has nothing to do.
 */
import type { FileStore } from '@/core/storage/files';
import type { BrowserStorage } from './browser-storage';

export class OpfsFileStore implements FileStore {
  readonly dataDir = '/qq';
  private readonly storage: BrowserStorage;

  constructor(storage: BrowserStorage) {
    this.storage = storage;
  }

  exists(path: string): Promise<boolean> {
    return this.storage.call({ op: 'exists', path });
  }

  copy(from: string, to: string): Promise<void> {
    return this.storage.call({ op: 'copy', from, to });
  }

  move(from: string, to: string): Promise<void> {
    return this.storage.call({ op: 'move', from, to });
  }

  remove(path: string): Promise<void> {
    return this.storage.call({ op: 'remove', path });
  }

  async ensureDir(_path: string): Promise<void> {}

  size(path: string): Promise<number | null> {
    return this.storage.call({ op: 'size', path });
  }

  /** Saves bytes (a download, the built-in story) as a file, replacing it. Web only. */
  write(path: string, bytes: Uint8Array): Promise<void> {
    return this.storage.call({ op: 'write', path, bytes });
  }
}
