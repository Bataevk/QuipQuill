// Types for db-host.mjs (plain JS, served past Metro).
import type { Row, SqlValue } from '@/core/storage/db';

export type StorageRequest =
  | { op: 'open'; path: string }
  | { op: 'close'; db: number }
  | { op: 'run'; db: number; sql: string; params: readonly SqlValue[] }
  | { op: 'all'; db: number; sql: string; params: readonly SqlValue[] }
  | { op: 'exec'; db: number; sql: string }
  | { op: 'exists'; path: string }
  | { op: 'size'; path: string }
  | { op: 'remove'; path: string }
  | { op: 'copy'; from: string; to: string }
  | { op: 'move'; from: string; to: string }
  | { op: 'write'; path: string; bytes: Uint8Array };

export interface StorageResults {
  open: number;
  close: void;
  run: { changes: number; lastInsertRowId: number };
  all: Row[];
  exec: void;
  exists: boolean;
  size: number | null;
  remove: void;
  copy: void;
  move: void;
  write: void;
}

/** The PoolUtil methods the host uses (installOpfsSAHPoolVfs, or a stand-in in tests). */
export interface PoolLike {
  OpfsSAHPoolDb: new (path: string) => unknown;
  getFileNames(): string[];
  getFileCount(): number;
  getCapacity(): number;
  addCapacity(n: number): Promise<number>;
  exportFile(path: string): Uint8Array;
  importDb(path: string, bytes: Uint8Array): number;
  unlink(path: string): boolean;
}

export function createHost(sqlite3: unknown, pool: PoolLike): (request: StorageRequest) => Promise<unknown>;
