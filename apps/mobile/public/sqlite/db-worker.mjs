// The browser's database worker (stage W.2, ADR 69): installs the opfs-sahpool VFS once and
// answers storage requests from src/platform/web/browser-storage.ts. Loaded as
// new Worker('/sqlite/db-worker.mjs', { type: 'module' }); sqlite3.mjs and sqlite3.wasm are
// copied next to it by scripts/copy-sqlite-wasm.mjs.
import sqlite3InitModule from './sqlite3.mjs';
import { createHost } from './db-host.mjs';

/** Slots for app.db, a few stories and saves, journals and staging copies; grown on demand. */
const INITIAL_CAPACITY = 32;

const ready = (async () => {
  const sqlite3 = await sqlite3InitModule();
  const pool = await sqlite3.installOpfsSAHPoolVfs({
    name: 'foolore',
    initialCapacity: INITIAL_CAPACITY,
  });
  await pool.reserveMinimumCapacity(INITIAL_CAPACITY);
  return createHost(sqlite3, pool);
})();

self.onmessage = async ({ data }) => {
  try {
    const handle = await ready;
    self.postMessage({ id: data.id, result: await handle(data.request) });
  } catch (error) {
    self.postMessage({
      id: data.id,
      error: { name: error?.name ?? 'Error', message: String(error?.message ?? error) },
    });
  }
};
