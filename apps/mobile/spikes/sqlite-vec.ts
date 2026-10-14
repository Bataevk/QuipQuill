/**
 * Spike 3.1a: op-sqlite with sqlite-vec on the device, and brute force in JS as the fallback.
 *
 * Mirrors what story import will do: vectors arrive as float32 BLOBs in an ordinary table
 * (as in .qqstory), the vec0 index is built from them, then a KNN query must find the right
 * entity. 5 000 × 1536-d vectors ≈ the upper bound of a story in the prototype (ADR 5).
 */
import { open } from '@op-engineering/op-sqlite';

const DIM = 1536;
const COUNT = 5000;
const TARGET = 42;
const QUERIES = 10;

export type SqliteSpikeResult = {
  ok: boolean;
  sqliteVersion?: string;
  vecVersion?: string;
  checks: Record<string, boolean>;
  ms: Record<string, number>;
  error?: string;
};

/** Deterministic PRNG so every run uses the same vectors. */
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Unit vectors, like text-embedding-3-small: L2 order equals cosine order. */
function unitVector(random: () => number): Float32Array {
  const v = new Float32Array(DIM);
  let norm = 0;
  for (let i = 0; i < DIM; i++) {
    v[i] = random() * 2 - 1;
    norm += v[i] * v[i];
  }
  norm = Math.sqrt(norm);
  for (let i = 0; i < DIM; i++) v[i] /= norm;
  return v;
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

export async function runSqliteSpike(): Promise<SqliteSpikeResult> {
  const checks: Record<string, boolean> = {};
  const ms: Record<string, number> = {};
  const result: SqliteSpikeResult = { ok: false, checks, ms };
  const db = open({ name: 'spike-3-1.db' });
  try {
    for (const table of ['vec_entities', 'embeddings', 'strict_check']) {
      await db.execute(`DROP TABLE IF EXISTS ${table}`);
    }

    const versions = await db.execute('SELECT sqlite_version() AS v, vec_version() AS vec');
    result.sqliteVersion = String(versions.rows[0].v);
    result.vecVersion = String(versions.rows[0].vec);

    // The .qqstory schema relies on STRICT tables and json_valid (SQLite ≥ 3.38).
    await db.execute(
      "CREATE TABLE strict_check (props TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(props))) STRICT",
    );
    await db.execute('INSERT INTO strict_check (props) VALUES (?)', ['{"fixed": true}']);
    let rejected = false;
    try {
      await db.execute('INSERT INTO strict_check (props) VALUES (?)', ['not json']);
    } catch {
      rejected = true;
    }
    checks.strictAndJsonValid = rejected;

    const random = mulberry32(7);
    const vectors = Array.from({ length: COUNT }, () => unitVector(random));

    let started = Date.now();
    await db.execute('CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vector BLOB NOT NULL)');
    await db.executeBatch([
      ['INSERT INTO embeddings (id, vector) VALUES (?, ?)', vectors.map((v, i) => [i + 1, v])],
    ]);
    ms.insertBlobs = Date.now() - started;

    // What import does: build the index from the BLOBs already in the story file.
    started = Date.now();
    await db.execute(`CREATE VIRTUAL TABLE vec_entities USING vec0(embedding float[${DIM}])`);
    await db.execute('INSERT INTO vec_entities (rowid, embedding) SELECT id, vector FROM embeddings');
    ms.buildVecIndex = Date.now() - started;

    // A query close to vector #TARGET (a little noise added).
    const query = new Float32Array(vectors[TARGET - 1]);
    for (let i = 0; i < DIM; i += 50) query[i] += 0.01;

    started = Date.now();
    let vecTop: number[] = [];
    for (let q = 0; q < QUERIES; q++) {
      const knn = await db.execute(
        'SELECT rowid, distance FROM vec_entities WHERE embedding MATCH ? ORDER BY distance LIMIT 5',
        [query],
      );
      vecTop = knn.rows.map((row) => Number(row.rowid));
    }
    ms.vecQueryAvg = (Date.now() - started) / QUERIES;
    checks.vecFindsTarget = vecTop[0] === TARGET;

    // Fallback: load every vector into JS and scan.
    started = Date.now();
    const rows = await db.execute('SELECT id, vector FROM embeddings');
    const loaded = rows.rows.map((row) => ({
      id: Number(row.id),
      vector: new Float32Array(row.vector as ArrayBuffer),
    }));
    ms.bruteLoad = Date.now() - started;

    started = Date.now();
    let bruteTop = -1;
    for (let q = 0; q < QUERIES; q++) {
      let best = -Infinity;
      for (const item of loaded) {
        const score = dot(query, item.vector);
        if (score > best) {
          best = score;
          bruteTop = item.id;
        }
      }
    }
    ms.bruteQueryAvg = (Date.now() - started) / QUERIES;
    checks.bruteFindsTarget = bruteTop === TARGET;

    result.ok = Object.values(checks).every(Boolean);
  } catch (error) {
    const e = error as Error;
    result.error = `${e.name}: ${e.message}`.slice(0, 2000);
  } finally {
    db.close();
  }
  return result;
}
