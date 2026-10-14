import { blob, type Db } from '../storage/db';
import type { EmbeddingKind, VectorHit, VectorIndex } from './vector-index';

const TABLE = 'vec_entities';

/**
 * KNN through sqlite-vec's `vec0` table, whose rowids are the rowids of `embeddings`.
 * Distances are L2; for unit vectors (text-embedding-3-small) cosine = 1 − d² / 2.
 */
export class SqliteVecIndex implements VectorIndex {
  readonly kind = 'sqlite-vec';
  private readonly db: Db;

  constructor(db: Db) {
    this.db = db;
  }

  /**
   * (Re)builds the index from the BLOBs in `embeddings`. Returns false if the sqlite-vec
   * extension is not loaded, so the caller can fall back to brute force.
   */
  static async build(db: Db, dim: number): Promise<boolean> {
    try {
      await db.exec(`DROP TABLE IF EXISTS ${TABLE}`);
      await db.exec(`CREATE VIRTUAL TABLE ${TABLE} USING vec0(embedding float[${dim}])`);
    } catch {
      return false;
    }
    await db.run(`INSERT INTO ${TABLE} (rowid, embedding) SELECT rowid, vector FROM embeddings`);
    return true;
  }

  /** True if the file has a built index and this connection can query it. */
  static async usable(db: Db): Promise<boolean> {
    try {
      await db.get(`SELECT rowid FROM ${TABLE} LIMIT 1`);
      return true;
    } catch {
      return false; // no table, or the extension is not loaded in this connection
    }
  }

  async search(query: Float32Array, limit: number): Promise<VectorHit[]> {
    if (limit <= 0) return [];
    const rows = await this.db.all<{ entity_id: string; kind: string; distance: number }>(
      `SELECT e.entity_id, e.kind, v.distance
         FROM (SELECT rowid, distance FROM ${TABLE} WHERE embedding MATCH ? AND k = ?) AS v
         JOIN embeddings AS e ON e.rowid = v.rowid
        ORDER BY v.distance`,
      [blob(query), limit],
    );
    return rows.map((row) => ({
      entityId: row.entity_id,
      kind: row.kind as EmbeddingKind,
      score: 1 - (Number(row.distance) ** 2) / 2,
    }));
  }
}
