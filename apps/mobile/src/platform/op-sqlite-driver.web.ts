/**
 * Stage W.1 placeholder: op-sqlite's own web backend cannot serve the core (no ROLLBACK, no
 * file export/import for FileStore, a 6-slot pool) and its bundled sqlite3-worker1 breaks the
 * Metro export, so the browser gets its own driver over SQLite WASM in stage W.2 (ADR 69).
 * This keeps op-sqlite out of the web bundle until then.
 */
import type { Db, DbDriver } from '@/core/storage/db';

export class OpSqliteDriver implements DbDriver {
  async open(): Promise<Db> {
    throw new Error('The browser database arrives in stage W.2.');
  }
}
