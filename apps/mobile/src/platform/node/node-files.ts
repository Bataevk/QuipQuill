/// <reference types="node" />
/** FileStore port on node:fs. Used by core tests (and later the desktop app). */
import { access, copyFile, mkdir, rename, rm } from 'node:fs/promises';

import type { FileStore } from '@/core/storage/files';

export class NodeFileStore implements FileStore {
  readonly dataDir: string;

  constructor(dataDir: string) {
    this.dataDir = dataDir;
  }

  async exists(path: string): Promise<boolean> {
    try {
      await access(path);
      return true;
    } catch {
      return false;
    }
  }

  async copy(from: string, to: string): Promise<void> {
    await copyFile(from, to);
  }

  async move(from: string, to: string): Promise<void> {
    await rename(from, to);
  }

  async remove(path: string): Promise<void> {
    await rm(path, { force: true });
  }

  async ensureDir(path: string): Promise<void> {
    await mkdir(path, { recursive: true });
  }
}
