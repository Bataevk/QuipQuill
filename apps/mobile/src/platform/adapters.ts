/**
 * This platform's adapters for the core's ports: op-sqlite and expo on iOS and Android. The
 * browser has its own set in adapters.web.ts (stage W.2, ADR 69); Metro picks the file.
 */
import type { TokenStore } from '@/core/net/auth';
import type { Downloader } from '@/core/net/download';
import type { DbDriver } from '@/core/storage/db';
import type { FileStore } from '@/core/storage/files';
import { ExpoDownloader } from './expo-downloader';
import { ExpoFileStore } from './expo-files';
import { OpSqliteDriver } from './op-sqlite-driver';
import { SecureTokenStore } from './secure-token-store';

export interface Adapters {
  driver: DbDriver;
  files: FileStore;
  downloader: Downloader;
  tokens: TokenStore;
}

let adapters: Adapters | undefined;

export function getAdapters(): Adapters {
  adapters ??= {
    driver: new OpSqliteDriver(),
    files: new ExpoFileStore(),
    downloader: new ExpoDownloader(),
    tokens: new SecureTokenStore(),
  };
  return adapters;
}
