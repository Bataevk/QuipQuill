/**
 * Vector search over a story's embeddings (ADR 26). Two adapters behind one port:
 * sqlite-vec (`vec0`, fast, needs the extension) and brute force in JS (works everywhere).
 */
export type EmbeddingKind = 'name' | 'desc';

export interface VectorHit {
  entityId: string;
  kind: EmbeddingKind;
  /** Cosine similarity, higher is closer. */
  score: number;
}

export type VectorIndexKind = 'sqlite-vec' | 'brute-force';

export interface VectorIndex {
  readonly kind: VectorIndexKind;
  /** The `limit` closest embeddings to `query` (same model and dimension as the story). */
  search(query: Float32Array, limit: number): Promise<VectorHit[]>;
}

/** Float32 little-endian BLOB (as builder writes it) → vector. Copies, so alignment is safe. */
export function toVector(bytes: Uint8Array): Float32Array {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}
