/**
 * Db port on op-sqlite (the app). sqlite-vec is compiled in (`"op-sqlite": {"sqliteVec": true}`
 * in package.json) and loaded into every connection. Read-only opening is not supported by
 * op-sqlite; the core never writes to a connection it opened read-only, so it is ignored.
 */
import { open, type DB, type QueryResult, type Scalar } from '@op-engineering/op-sqlite';

import type { Db, DbDriver, Row, RunResult, SqlParams, SqlValue } from '@/core/storage/db';

export class OpSqliteDriver implements DbDriver {
  async open(path: string): Promise<Db> {
    const plain = path.startsWith('file://') ? decodeURIComponent(path.slice(7)) : path;
    const slash = plain.lastIndexOf('/');
    return new OpSqliteDb(open({ name: plain.slice(slash + 1), location: plain.slice(0, slash) }));
  }
}

type Execute = (sql: string, params?: Scalar[]) => Promise<QueryResult>;

class OpSqliteDb implements Db {
  private readonly db: DB;
  private readonly execute: Execute;

  constructor(db: DB, execute?: Execute) {
    this.db = db;
    this.execute = execute ?? ((sql, params) => db.execute(sql, params));
  }

  async run(sql: string, params: SqlParams = []): Promise<RunResult> {
    const result = await this.execute(sql, params as Scalar[]);
    return { changes: result.rowsAffected, lastInsertRowId: result.insertId ?? 0 };
  }

  async all<T extends Row = Row>(sql: string, params: SqlParams = []): Promise<T[]> {
    const result = await this.execute(sql, params as Scalar[]);
    return result.rows.map((row) => normalize(row) as T);
  }

  async get<T extends Row = Row>(sql: string, params: SqlParams = []): Promise<T | undefined> {
    return (await this.all<T>(sql, params))[0];
  }

  async exec(script: string): Promise<void> {
    await this.execute(script); // op-sqlite runs every statement in the string
  }

  async transaction<T>(fn: (tx: Db) => Promise<T>): Promise<T> {
    let result!: T;
    // op-sqlite queues transactions on a connection and rolls back when the callback throws.
    await this.db.transaction(async (tx) => {
      result = await fn(new OpSqliteDb(this.db, (sql, params) => tx.execute(sql, params)));
    });
    return result;
  }

  async close(): Promise<void> {
    this.db.close();
  }
}

/** BLOBs come back as ArrayBuffer; the port promises Uint8Array. */
function normalize(row: Record<string, Scalar>): Row {
  const out: Row = {};
  for (const [key, value] of Object.entries(row)) {
    out[key] = value instanceof ArrayBuffer ? new Uint8Array(value) : (value as SqlValue);
  }
  return out;
}
