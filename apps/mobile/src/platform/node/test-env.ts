/// <reference types="node" />
/** Test environment for core tests: a temp data dir, node:sqlite with sqlite-vec, the library. */
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { getLoadablePath } from 'sqlite-vec';

import { Library } from '@/core/library/library';
import { NodeFileStore } from './node-files';
import { NodeSqliteDriver } from './node-sqlite-driver';

/** The demo story built by builder (stage 1.3) — the fixture the roadmap asks for. */
export const DEMO_STORY = fileURLToPath(
  new URL('../../../../../stories/brass-vault/brass-vault.qqstory', import.meta.url),
);
export const DEMO_STORY_ID = 'd09514c5-6b96-48a4-ba57-2cc4b4b68f55';

export interface TestEnv {
  dir: string;
  files: NodeFileStore;
  driver: NodeSqliteDriver;
  library: Library;
  cleanup(): Promise<void>;
}

/** `vec: false` simulates a platform without the sqlite-vec extension. */
export async function makeTestEnv(options: { vec?: boolean } = {}): Promise<TestEnv> {
  const dir = await mkdtemp(join(tmpdir(), 'qq-core-'));
  const files = new NodeFileStore(join(dir, 'data'));
  const driver = new NodeSqliteDriver({
    extensions: options.vec === false ? [] : [getLoadablePath()],
  });
  const library = await Library.open(driver, files);
  return {
    dir,
    files,
    driver,
    library,
    async cleanup() {
      await library.close();
      await rm(dir, { recursive: true, force: true });
    },
  };
}

/**
 * A stand-in for the story's embedding model: a fixed pseudo-random unit vector per text, so a
 * query made from the same text matches it exactly (score 1) and other texts barely at all.
 */
export function fakeEmbedding(text: string, dim = 1536): Float32Array {
  let seed = 2166136261;
  for (let i = 0; i < text.length; i++) seed = Math.imul(seed ^ text.charCodeAt(i), 16777619) >>> 0;
  const vector = new Float32Array(dim);
  let sum = 0;
  for (let i = 0; i < dim; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    vector[i] = seed / 2 ** 32 - 0.5;
    sum += vector[i] * vector[i];
  }
  const norm = Math.sqrt(sum);
  return vector.map((x) => x / norm);
}
