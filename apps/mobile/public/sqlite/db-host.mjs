// The browser's storage behind the core's Db and FileStore ports (stage W.2, ADR 69): SQLite
// WASM on the opfs-sahpool VFS. Plain JS served from public/sqlite/ next to sqlite3.mjs, past
// Metro (spike W.1). db-worker.mjs runs it in the worker; Node tests run it over a stand-in pool.
//
// Every file lives in the pool under its absolute path. The pool only holds SQLite databases, so
// anything else written here (a refused or cut download's body) is kept in memory as a "loose"
// file: the core still sees its size, as on the phone, and nothing invalid reaches OPFS.

/** Room for journals and staging copies before the pool is grown. */
const SPARE_SLOTS = 4;
const GROW_BY = 16;
const HEADER = 'SQLite format 3';

function isDatabase(bytes) {
  if (bytes.byteLength < 512 || bytes.byteLength % 512 !== 0) return false;
  for (let i = 0; i < HEADER.length; i++) if (bytes[i] !== HEADER.charCodeAt(i)) return false;
  return true;
}

/**
 * `pool` is the PoolUtil of installOpfsSAHPoolVfs (or a stand-in with the same methods).
 * Returns `handle(request) → result`; requests run one at a time, in arrival order.
 */
export function createHost(sqlite3, pool) {
  const dbs = new Map();
  const loose = new Map();
  let nextDb = 1;
  let tail = Promise.resolve();

  const inPool = (path) => pool.getFileNames().includes(path);

  async function ensureRoom() {
    if (pool.getFileCount() + SPARE_SLOTS >= pool.getCapacity()) await pool.addCapacity(GROW_BY);
  }

  function db(id) {
    const open = dbs.get(id);
    if (!open) throw new Error(`The database connection ${id} is closed.`);
    return open;
  }

  function read(path) {
    if (loose.has(path)) return loose.get(path);
    if (inPool(path)) return pool.exportFile(path);
    throw new Error(`No such file: ${path}`);
  }

  function remove(path) {
    loose.delete(path);
    if (inPool(path)) pool.unlink(path);
  }

  async function write(path, bytes) {
    // importDb overwrites in place without truncating: a shorter file would keep the old tail.
    remove(path);
    if (!isDatabase(bytes)) {
      loose.set(path, bytes);
      return;
    }
    await ensureRoom();
    pool.importDb(path, bytes);
  }

  const ops = {
    async open({ path }) {
      if (loose.has(path)) throw new Error(`${path} is not a database.`);
      if (!inPool(path)) await ensureRoom();
      const id = nextDb++;
      dbs.set(id, new pool.OpfsSAHPoolDb(path));
      return id;
    },
    close({ db: id }) {
      dbs.get(id)?.close();
      dbs.delete(id);
    },
    run({ db: id, sql, params }) {
      const conn = db(id);
      conn.exec({ sql, bind: params.length > 0 ? params : undefined });
      return {
        changes: conn.changes(),
        lastInsertRowId: Number(sqlite3.capi.sqlite3_last_insert_rowid(conn.pointer)),
      };
    },
    all({ db: id, sql, params }) {
      return db(id).exec({
        sql,
        bind: params.length > 0 ? params : undefined,
        rowMode: 'object',
        returnValue: 'resultRows',
      });
    },
    exec({ db: id, sql }) {
      db(id).exec(sql);
    },
    exists({ path }) {
      if (loose.has(path) || inPool(path)) return true;
      // No directories in the pool: a "directory" exists while some file lies under it.
      const prefix = `${path.replace(/\/+$/, '')}/`;
      return pool.getFileNames().some((name) => name.startsWith(prefix));
    },
    size({ path }) {
      if (loose.has(path)) return loose.get(path).byteLength;
      return inPool(path) ? pool.exportFile(path).byteLength : null;
    },
    remove({ path }) {
      remove(path);
    },
    async copy({ from, to }) {
      await write(to, read(from));
    },
    async move({ from, to }) {
      if (from === to) return;
      await write(to, read(from));
      remove(from);
    },
    async write({ path, bytes }) {
      await write(path, bytes);
    },
  };

  return function handle(request) {
    const op = ops[request.op];
    const run = tail.then(() => {
      if (!op) throw new Error(`Unknown storage operation: ${request.op}`);
      return op(request);
    });
    tail = run.catch(() => undefined);
    return run;
  };
}
