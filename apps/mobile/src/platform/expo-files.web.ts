/**
 * Stage W.1 placeholder: expo-file-system has no browser backend (`Paths.document` throws), so
 * the browser gets a FileStore over the SQLite WASM pool in stage W.2 (ADR 69). Same exports as
 * expo-files.ts; every call fails with a clear error instead of breaking module load.
 */
import type { File } from 'expo-file-system';

import type { FileStore } from '@/core/storage/files';

const notYet = (): never => {
  throw new Error('Browser storage arrives in stage W.2.');
};

export class ExpoFileStore implements FileStore {
  readonly dataDir = '/qq';
  exists = async (): Promise<boolean> => notYet();
  copy = async (): Promise<void> => notYet();
  move = async (): Promise<void> => notYet();
  remove = async (): Promise<void> => notYet();
  ensureDir = async (): Promise<void> => notYet();
  size = async (): Promise<number | null> => notYet();
}

export const toPath = (uri: string): string => uri;
export const toUri = (path: string): string => path;
export const pickedFilePath = async (_file: File): Promise<{ path: string; temporary: boolean }> => notYet();
export const removeFile = (_path: string): void => notYet();
