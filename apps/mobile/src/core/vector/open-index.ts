import type { Db } from '../storage/db';
import { BruteForceIndex } from './brute-force-index';
import { SqliteVecIndex } from './sqlite-vec-index';
import type { VectorIndex } from './vector-index';

/** sqlite-vec when the file has an index this connection can use, brute force otherwise. */
export async function openVectorIndex(db: Db): Promise<VectorIndex> {
  return (await SqliteVecIndex.usable(db)) ? new SqliteVecIndex(db) : new BruteForceIndex(db);
}
