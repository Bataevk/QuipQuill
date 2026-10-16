import type { Db } from '../storage/db';
import { toVector, type EmbeddingKind, type VectorHit, type VectorIndex } from './vector-index';

export interface Entry {
  entityId: string;
  kind: EmbeddingKind;
  vector: Float32Array;
  norm: number;
}

/** Loads every embedding once, then scans: ~0.2 s per query for 5k × 1536 (spike 3.1a). */
export class BruteForceIndex implements VectorIndex {
  readonly kind = 'brute-force';
  private readonly db: Db;
  private entries: Promise<Entry[]> | undefined;

  constructor(db: Db) {
    this.db = db;
  }

  async search(query: Float32Array, limit: number): Promise<VectorHit[]> {
    return rankByCosine(await (this.entries ??= this.load()), query, limit);
  }

  private async load(): Promise<Entry[]> {
    const rows = await this.db.all<{ entity_id: string; kind: string; vector: Uint8Array }>(
      'SELECT entity_id, kind, vector FROM embeddings',
    );
    return rows.map((row) => toEntry(row.entity_id, row.kind as EmbeddingKind, toVector(row.vector)));
  }
}

export function toEntry(entityId: string, kind: EmbeddingKind, vector: Float32Array): Entry {
  return { entityId, kind, vector, norm: norm(vector) };
}

/** The `limit` entries closest to `query` by cosine similarity (a plain scan). */
export function rankByCosine(entries: readonly Entry[], query: Float32Array, limit: number): VectorHit[] {
  const queryNorm = norm(query);
  if (queryNorm === 0 || limit <= 0) return [];
  const hits = entries
    .filter((entry) => entry.vector.length === query.length && entry.norm > 0)
    .map((entry) => ({
      entityId: entry.entityId,
      kind: entry.kind,
      score: dot(query, entry.vector) / (queryNorm * entry.norm),
    }));
  hits.sort((a, b) => b.score - a.score);
  return hits.slice(0, limit);
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function norm(v: Float32Array): number {
  return Math.sqrt(dot(v, v));
}
