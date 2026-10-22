/**
 * Db port on SQLite WASM in the browser's storage worker (stage W.2, ADR 69). Read-only opening
 * is not supported by the pool; as with op-sqlite, the core never writes to a connection it
 * opened read-only, so the option is ignored.
 */
import type { Db, DbDriver, Row, RunResult, SqlParams } from '@/core/storage/db';
import type { BrowserStorage } from './browser-storage';

export class WasmSqliteDriver implements DbDriver {
  private readonly storage: BrowserStorage;

  constructor(storage: BrowserStorage) {
    this.storage = storage;
  }

  async open(path: string): Promise<Db> {
    return new WasmSqliteDb(this.storage, await this.storage.call({ op: 'open', path }));
  }
}

class WasmSqliteDb implements Db {
  private readonly storage: BrowserStorage;
  private readonly id: number;
  private queue: Promise<unknown> = Promise.resolve();
  private closed = false;

  constructor(storage: BrowserStorage, id: number) {
    this.storage = storage;
    this.id = id;
  }

  run(sql: string, params: SqlParams = []): Promise<RunResult> {
    return this.storage.call({ op: 'run', db: this.id, sql, params });
  }

  all<T extends Row = Row>(sql: string, params: SqlParams = []): Promise<T[]> {
    return this.storage.call({ op: 'all', db: this.id, sql, params }) as Promise<T[]>;
  }

  async get<T extends Row = Row>(sql: string, params: SqlParams = []): Promise<T | undefined> {
    return (await this.all<T>(sql, params))[0];
  }

  exec(script: string): Promise<void> {
    return this.storage.call({ op: 'exec', db: this.id, sql: script });
  }

  transaction<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    // Like node-sqlite-driver: transactions on a connection queue up; a failed one does not
    // block the next. The worker runs one request at a time, in order.
    const run = this.queue.then(async () => {
      await this.exec('BEGIN');
      try {
        const result = await fn(this);
        await this.exec('COMMIT');
        return result;
      } catch (error) {
        // SQLite may have rolled back on its own already; the caller needs the first error.
        await this.exec('ROLLBACK').catch(() => {});
        throw error;
      }
    });
    this.queue = run.catch(() => undefined);
    return run;
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    await this.storage.call({ op: 'close', db: this.id });
  }
}
