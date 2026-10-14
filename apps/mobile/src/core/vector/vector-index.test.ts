import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { Db } from '@/core/storage/db';
import { BruteForceIndex } from '@/core/vector/brute-force-index';
import { openVectorIndex } from '@/core/vector/open-index';
import { SqliteVecIndex } from '@/core/vector/sqlite-vec-index';
import { toVector } from '@/core/vector/vector-index';
import { DEMO_STORY, makeTestEnv, type TestEnv } from '@/platform/node/test-env';

let env: TestEnv;
let db: Db;

afterEach(async () => {
  await db?.close();
  await env?.cleanup();
});

async function openImported(options: { vec?: boolean } = {}): Promise<Db> {
  env = await makeTestEnv(options);
  const { story } = await env.library.importer.import(DEMO_STORY);
  db = await env.driver.open(join(env.files.dataDir, story.path), { readOnly: true });
  return db;
}

async function embeddingOf(db: Db, entityId: string, kind: string): Promise<Float32Array> {
  const row = await db.get<{ vector: Uint8Array }>(
    'SELECT vector FROM embeddings WHERE entity_id = ? AND kind = ?',
    [entityId, kind],
  );
  return toVector(row!.vector);
}

describe('vector indexes', () => {
  it('sqlite-vec and brute force agree on the nearest embeddings', async () => {
    await openImported();
    const query = await embeddingOf(db, 'npc:bertha-koss', 'desc');

    const vec = await new SqliteVecIndex(db).search(query, 5);
    const brute = await new BruteForceIndex(db).search(query, 5);

    expect(vec[0]).toMatchObject({ entityId: 'npc:bertha-koss', kind: 'desc' });
    expect(vec[0].score).toBeCloseTo(1, 5);
    expect(vec.map((hit) => [hit.entityId, hit.kind])).toEqual(
      brute.map((hit) => [hit.entityId, hit.kind]),
    );
    // vec0 turns L2 into cosine assuming unit vectors; real embeddings are unit only to ~1e-4.
    for (let i = 0; i < vec.length; i++) expect(vec[i].score).toBeCloseTo(brute[i].score, 3);
  });

  it('opens sqlite-vec when usable and brute force otherwise', async () => {
    await openImported();
    expect((await openVectorIndex(db)).kind).toBe('sqlite-vec');
    await db.close();
    await env.cleanup();

    await openImported({ vec: false });
    expect((await openVectorIndex(db)).kind).toBe('brute-force');
  });

  it('returns nothing for a zero limit', async () => {
    await openImported();
    const query = await embeddingOf(db, 'npc:bertha-koss', 'name');
    expect(await new SqliteVecIndex(db).search(query, 0)).toEqual([]);
    expect(await new BruteForceIndex(db).search(query, 0)).toEqual([]);
  });
});
