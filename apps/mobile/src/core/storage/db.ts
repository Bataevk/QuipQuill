/**
 * Database port. The core talks SQL through this interface only; adapters live in the platform
 * layer: op-sqlite in the app (src/platform/op-sqlite-driver.ts), node:sqlite in Node tests and
 * later on desktop (src/platform/node/node-sqlite-driver.ts).
 */

/** BLOBs travel as Uint8Array in both directions, whatever the driver uses internally. */
export type SqlValue = string | number | bigint | null | Uint8Array;
export type SqlParams = readonly SqlValue[];
export type Row = Record<string, SqlValue>;

export interface RunResult {
  changes: number;
  lastInsertRowId: number;
}

export interface Db {
  /** A statement that returns no rows (INSERT, UPDATE, DDL). */
  run(sql: string, params?: SqlParams): Promise<RunResult>;
  all<T extends Row = Row>(sql: string, params?: SqlParams): Promise<T[]>;
  get<T extends Row = Row>(sql: string, params?: SqlParams): Promise<T | undefined>;
  /** Several `;`-separated statements without parameters (schema scripts). */
  exec(script: string): Promise<void>;
  /**
   * Runs `fn` atomically: COMMIT when it resolves, ROLLBACK when it throws. Use the `tx`
   * handle inside; do not start queries on the outer handle meanwhile. Not reentrant.
   */
  transaction<T>(fn: (tx: Db) => Promise<T>): Promise<T>;
  close(): Promise<void>;
}

export interface OpenOptions {
  readOnly?: boolean;
}

export interface DbDriver {
  /** `path` is a plain file path (no `file://`), absolute or relative to the process. */
  open(path: string, options?: OpenOptions): Promise<Db>;
}

/** Bytes of a typed array, e.g. a Float32Array vector, as a BLOB parameter. */
export function blob(view: ArrayBufferView): Uint8Array {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

/** Opens, runs `fn`, and always closes. */
export async function withDb<T>(
  driver: DbDriver,
  path: string,
  fn: (db: Db) => Promise<T>,
  options?: OpenOptions,
): Promise<T> {
  const db = await driver.open(path, options);
  try {
    return await fn(db);
  } finally {
    await db.close();
  }
}
