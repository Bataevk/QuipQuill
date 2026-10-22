/// <reference types="node" />
/**
 * The browser's storage adapters (stage W.2) in Node: the real SQLite WASM build and the worker's
 * host logic (public/sqlite/db-host.mjs), over an in-memory stand-in for the opfs-sahpool pool
 * (OPFS exists only in browsers). Messages are cloned as postMessage would clone them.
 */
import { readFile } from 'node:fs/promises';

import sqlite3InitModule from '@sqlite.org/sqlite-wasm';
import { beforeEach, describe, expect, it } from 'vitest';

import { Library } from '@/core/library/library';
import { blob } from '@/core/storage/db';
import { DEMO_STORY, DEMO_STORY_ID } from '@/platform/node/test-env';
import { createHost, type PoolLike } from '../../../public/sqlite/db-host.mjs';
import { BrowserStorage, StorageBusyError, type WorkerPort } from './browser-storage';
import { OpfsFileStore } from './opfs-files';
import { WasmSqliteDriver } from './wasm-sqlite-driver';

type Sqlite3 = Awaited<ReturnType<typeof sqlite3InitModule>>;
type MemDb = InstanceType<Sqlite3['oo1']['DB']>;

const sqlite3: Sqlite3 = await sqlite3InitModule();

/**
 * The pool's behaviour the host relies on: files by name, databases opened by name, export and
 * import of whole files. Bytes are kept between connections, as OPFS keeps them between reloads.
 */
function memoryPool(capacity = 6): PoolLike & { capacity: number } {
  const files = new Map<string, Uint8Array>();
  const open = new Map<string, MemDb>();
  const snapshot = (db: MemDb) => sqlite3.capi.sqlite3_js_db_export(db);

  class OpfsSAHPoolDb extends sqlite3.oo1.DB {
    private readonly path: string;
    constructor(path: string) {
      super(':memory:');
      this.path = path;
      const bytes = files.get(path);
      if (bytes) {
        const p = sqlite3.wasm.allocFromTypedArray(bytes);
        const flags = sqlite3.capi.SQLITE_DESERIALIZE_FREEONCLOSE | sqlite3.capi.SQLITE_DESERIALIZE_RESIZEABLE;
        sqlite3.capi.sqlite3_deserialize(this.pointer!, 'main', p, bytes.byteLength, bytes.byteLength, flags);
      }
      files.set(path, snapshot(this));
      open.set(path, this);
    }
    close() {
      if (open.get(this.path) === this) {
        files.set(this.path, snapshot(this));
        open.delete(this.path);
      }
      super.close();
    }
  }

  return {
    capacity,
    OpfsSAHPoolDb,
    getFileNames: () => [...files.keys()],
    getFileCount: () => files.size,
    getCapacity() {
      return this.capacity;
    },
    async addCapacity(n) {
      this.capacity += n;
      return this.capacity;
    },
    exportFile(path) {
      const db = open.get(path);
      const bytes = db ? snapshot(db) : files.get(path);
      if (!bytes) throw new Error(`File not found: ${path}`);
      return bytes;
    },
    importDb(path, bytes) {
      files.set(path, new Uint8Array(bytes));
      return bytes.byteLength;
    },
    unlink: (path) => files.delete(path),
  };
}

/** A worker port wired straight to the host, with structured cloning both ways. */
function directPort(pool: PoolLike): WorkerPort {
  const handle = createHost(sqlite3, pool);
  const port: WorkerPort = {
    onmessage: null,
    onerror: null,
    postMessage(message) {
      const { id, request } = structuredClone(message) as { id: number; request: Parameters<typeof handle>[0] };
      void handle(request).then(
        (result) => port.onmessage?.({ data: structuredClone({ id, result }) }),
        (error: Error) => port.onmessage?.({ data: { id, error: { name: error.name, message: error.message } } }),
      );
    },
  };
  return port;
}

function adapters(pool: PoolLike) {
  const storage = new BrowserStorage(async () => directPort(pool));
  return { storage, driver: new WasmSqliteDriver(storage), files: new OpfsFileStore(storage) };
}

let pool: ReturnType<typeof memoryPool>;
beforeEach(() => {
  pool = memoryPool();
});

describe('WasmSqliteDriver', () => {
  it('runs statements and reads rows, BLOBs and big integers included', async () => {
    const { driver } = adapters(pool);
    const db = await driver.open('/qq/test.db');
    await db.exec('CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, v BLOB, n INTEGER); CREATE TABLE u (x);');
    const vector = new Float32Array([0.25, -1.5, 3.125]);
    const first = await db.run('INSERT INTO t (name, v, n) VALUES (?, ?, ?)', ['a', blob(vector), 9007199254740993n]);
    expect(first).toEqual({ changes: 1, lastInsertRowId: 1 });
    await db.run('INSERT INTO t (name) VALUES (?)', ['b']);

    const row = await db.get<{ name: string; v: Uint8Array; n: bigint }>('SELECT name, v, n FROM t WHERE id = ?', [1]);
    expect(row?.name).toBe('a');
    expect(row?.v).toBeInstanceOf(Uint8Array);
    expect(Array.from(new Float32Array(row!.v.slice().buffer))).toEqual(Array.from(vector));
    expect(row?.n).toBe(9007199254740993n);
    expect(await db.all('SELECT name FROM t ORDER BY id')).toEqual([{ name: 'a' }, { name: 'b' }]);
    expect(await db.get('SELECT name FROM t WHERE id = ?', [99])).toBeUndefined();
    await db.close();
    await db.close(); // closing twice is harmless
  });

  it('commits a transaction and rolls back a failed one with its own error', async () => {
    const { driver } = adapters(pool);
    const db = await driver.open('/qq/tx.db');
    await db.exec('CREATE TABLE t (x INTEGER NOT NULL)');
    await db.transaction(async (tx) => {
      await tx.run('INSERT INTO t VALUES (1)');
    });
    await expect(
      db.transaction(async (tx) => {
        await tx.run('INSERT INTO t VALUES (2)');
        throw new Error('boom');
      }),
    ).rejects.toThrow('boom');
    await expect(db.transaction((tx) => tx.run('INSERT INTO t VALUES (NULL)'))).rejects.toThrow(/NOT NULL/);
    expect(await db.all('SELECT x FROM t')).toEqual([{ x: 1 }]);
  });

  it('queues transactions on one connection', async () => {
    const { driver } = adapters(pool);
    const db = await driver.open('/qq/queue.db');
    await db.exec('CREATE TABLE t (x)');
    const order: number[] = [];
    await Promise.all(
      [1, 2, 3].map((x) =>
        db.transaction(async (tx) => {
          await tx.run('INSERT INTO t VALUES (?)', [x]);
          order.push(x);
        }),
      ),
    );
    expect(order).toEqual([1, 2, 3]);
    expect((await db.get<{ n: number }>('SELECT count(*) AS n FROM t'))?.n).toBe(3);
  });

  it('reports SQL errors as rejections and keeps working', async () => {
    const { driver } = adapters(pool);
    const db = await driver.open('/qq/err.db');
    await expect(db.all('SELECT * FROM missing')).rejects.toThrow(/no such table/);
    expect(await db.get('SELECT 1 AS one')).toEqual({ one: 1 });
  });

  it('grows the pool before it runs out of slots', async () => {
    const { driver } = adapters(pool);
    for (let i = 0; i < 8; i++) await (await driver.open(`/qq/many-${i}.db`)).close();
    expect(pool.getCapacity()).toBeGreaterThan(8);
  });
});

describe('OpfsFileStore', () => {
  async function database(files: OpfsFileStore, driver: WasmSqliteDriver, path: string, rows: number) {
    const db = await driver.open(path);
    await db.exec('CREATE TABLE t (x)');
    for (let i = 0; i < rows; i++) await db.run('INSERT INTO t VALUES (?)', [new Uint8Array(1000)]);
    await db.close();
    return (await files.size(path))!;
  }

  it('copies, moves and removes files; directories exist through their files', async () => {
    const { files, driver } = adapters(pool);
    const size = await database(files, driver, '/qq/a.db', 3);
    expect(files.dataDir).toBe('/qq');
    expect(await files.exists('/qq/a.db')).toBe(true);
    expect(await files.exists('/qq')).toBe(true);
    expect(await files.exists('/qq/saves')).toBe(false);
    await files.ensureDir('/qq/saves');

    await files.copy('/qq/a.db', '/qq/saves/b.db');
    expect(await files.size('/qq/saves/b.db')).toBe(size);
    expect(await files.exists('/qq/saves')).toBe(true);

    await files.move('/qq/saves/b.db', '/qq/c.db');
    expect(await files.exists('/qq/saves/b.db')).toBe(false);
    const db = await driver.open('/qq/c.db');
    expect((await db.get<{ n: number }>('SELECT count(*) AS n FROM t'))?.n).toBe(3);
    await db.close();

    await files.remove('/qq/c.db');
    await files.remove('/qq/c.db'); // already gone: no error
    expect(await files.exists('/qq/c.db')).toBe(false);
    expect(await files.size('/qq/c.db')).toBeNull();
  });

  it('replaces a bigger file completely', async () => {
    const { files, driver } = adapters(pool);
    await database(files, driver, '/qq/big.db', 40);
    const small = await database(files, driver, '/qq/small.db', 1);
    await files.copy('/qq/small.db', '/qq/big.db');
    expect(await files.size('/qq/big.db')).toBe(small);
  });

  it('keeps bytes that are not a database in memory, with their size', async () => {
    const { files, driver } = adapters(pool);
    await files.write('/qq/downloads/x.part', new TextEncoder().encode('{"detail":"nope"}'));
    expect(await files.size('/qq/downloads/x.part')).toBe(17);
    expect(pool.getFileNames()).not.toContain('/qq/downloads/x.part');
    await expect(driver.open('/qq/downloads/x.part')).rejects.toThrow(/not a database/);
    await files.write('/qq/downloads/x.part', new Uint8Array());
    expect(await files.size('/qq/downloads/x.part')).toBe(0);
    await files.remove('/qq/downloads/x.part');
    expect(await files.exists('/qq/downloads/x.part')).toBe(false);
  });
});

describe('the core on browser storage', () => {
  it('imports the demo story, starts a game and finds it again after a reload', async () => {
    const first = adapters(pool);
    const library = await Library.open(first.driver, first.files);
    await first.files.write('/qq/bundled/brass-vault.qqstory', new Uint8Array(await readFile(DEMO_STORY)));
    const imported = await library.importer.importIfChanged('/qq/bundled/brass-vault.qqstory');
    expect(imported?.story.id).toBe(DEMO_STORY_ID);
    expect(imported?.story.vectorIndex).toBe('brute-force'); // no sqlite-vec in the browser (ADR 26)
    expect(await library.importer.importIfChanged('/qq/bundled/brass-vault.qqstory')).toBeUndefined();

    const save = await library.saves.create({ storyId: DEMO_STORY_ID, characterId: null, title: 'Test run' });
    const game = await library.saves.open(save.id);
    expect(await game.get("SELECT 1 AS ok FROM sqlite_master WHERE name = 'events'")).toEqual({ ok: 1 });
    await game.close();
    await library.close();

    // A reload: a new worker over the same pool.
    const second = adapters(pool);
    const reopened = await Library.open(second.driver, second.files);
    expect((await reopened.repository.listSaves()).map((s) => s.id)).toEqual([save.id]);
    expect((await reopened.storyMeta(DEMO_STORY_ID))?.title).toBe(imported?.meta.title);
    await reopened.saves.delete(save.id);
    expect(await second.files.exists(`/qq/${save.path}`)).toBe(false);
    await reopened.close();
  });
});

describe('BrowserStorage', () => {
  it('passes on a busy tab and tries again on the next call', async () => {
    let busy = true;
    const storage = new BrowserStorage(async () => {
      if (busy) throw new StorageBusyError();
      return directPort(pool);
    });
    const files = new OpfsFileStore(storage);
    await expect(files.exists('/qq/x')).rejects.toBeInstanceOf(StorageBusyError);
    busy = false;
    expect(await files.exists('/qq/x')).toBe(false);
  });

  it('fails pending and later calls when the worker breaks', async () => {
    const port: WorkerPort = { onmessage: null, onerror: null, postMessage: () => {} };
    const storage = new BrowserStorage(async () => port);
    const files = new OpfsFileStore(storage);
    const pending = files.exists('/qq/x');
    await Promise.resolve();
    await Promise.resolve();
    port.onerror?.({ message: 'wasm failed to load' });
    await expect(pending).rejects.toThrow(/wasm failed to load/);
    await expect(files.size('/qq/x')).rejects.toThrow(/wasm failed to load/);
  });
});
