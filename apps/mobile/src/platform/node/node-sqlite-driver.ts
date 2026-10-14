/// <reference types="node" />
/**
 * Db port on node:sqlite (built into Node ≥ 22, and into Electron for the desktop app).
 * Used by core tests; never imported by the React Native app.
 */
import { DatabaseSync, type SQLInputValue } from 'node:sqlite';

import type { Db, DbDriver, OpenOptions, Row, RunResult, SqlParams } from '@/core/storage/db';

export interface NodeSqliteOptions {
  /** Loadable extensions (e.g. sqlite-vec's getLoadablePath()) for every connection. */
  extensions?: string[];
}

export class NodeSqliteDriver implements DbDriver {
  private readonly extensions: string[];

  constructor(options: NodeSqliteOptions = {}) {
    this.extensions = options.extensions ?? [];
  }

  async open(path: string, options: OpenOptions = {}): Promise<Db> {
    const db = new DatabaseSync(path, {
      readOnly: options.readOnly ?? false,
      allowExtension: this.extensions.length > 0,
    });
    for (const extension of this.extensions) db.loadExtension(extension);
    return new NodeSqliteDb(db);
  }
}

class NodeSqliteDb implements Db {
  private readonly db: DatabaseSync;
  private queue: Promise<unknown> = Promise.resolve();

  constructor(db: DatabaseSync) {
    this.db = db;
  }

  async run(sql: string, params: SqlParams = []): Promise<RunResult> {
    const result = this.db.prepare(sql).run(...toInput(params));
    return { changes: Number(result.changes), lastInsertRowId: Number(result.lastInsertRowid) };
  }

  async all<T extends Row = Row>(sql: string, params: SqlParams = []): Promise<T[]> {
    return this.db.prepare(sql).all(...toInput(params)) as T[];
  }

  async get<T extends Row = Row>(sql: string, params: SqlParams = []): Promise<T | undefined> {
    return this.db.prepare(sql).get(...toInput(params)) as T | undefined;
  }

  async exec(script: string): Promise<void> {
    this.db.exec(script);
  }

  transaction<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    // Serialize transactions on this connection; a failed one must not block the next.
    const run = this.queue.then(async () => {
      this.db.exec('BEGIN');
      try {
        const result = await fn(this);
        this.db.exec('COMMIT');
        return result;
      } catch (error) {
        this.db.exec('ROLLBACK');
        throw error;
      }
    });
    this.queue = run.catch(() => undefined);
    return run;
  }

  async close(): Promise<void> {
    if (this.db.isOpen) this.db.close();
  }
}

function toInput(params: SqlParams): SQLInputValue[] {
  return params as SQLInputValue[];
}
