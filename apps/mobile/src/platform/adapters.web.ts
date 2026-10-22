/**
 * The browser's adapters for the core's ports (stage W.2, ADR 69): SQLite WASM on OPFS in one
 * worker behind the database and the files, fetch for downloads, localStorage for tokens. The
 * worker starts on the first call, once this tab holds the storage (one tab at a time).
 */
import { BrowserStorage, connectWorker } from './web/browser-storage';
import { FetchDownloader } from './web/fetch-downloader';
import { LocalTokenStore } from './web/local-token-store';
import { OpfsFileStore } from './web/opfs-files';
import { WasmSqliteDriver } from './web/wasm-sqlite-driver';
import type { Adapters } from './adapters';

export type { Adapters } from './adapters';

let adapters: (Adapters & { files: OpfsFileStore }) | undefined;

export function getAdapters(): Adapters & { files: OpfsFileStore } {
  if (!adapters) {
    const storage = new BrowserStorage(connectWorker);
    const files = new OpfsFileStore(storage);
    adapters = {
      driver: new WasmSqliteDriver(storage),
      files,
      downloader: new FetchDownloader(files),
      tokens: new LocalTokenStore(),
    };
  }
  return adapters;
}
