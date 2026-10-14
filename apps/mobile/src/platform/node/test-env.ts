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
