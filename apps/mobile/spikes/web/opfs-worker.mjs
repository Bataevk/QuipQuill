// Stage W.1 spike: SQLite WASM on the opfs-sahpool VFS in a module worker. Served from
// public/sqlite/ (copy it there next to sqlite3.mjs; see spikes/web/README.md).
// One message in ({story?: ArrayBuffer}), one report out ({ok, report} or {ok: false, error}).
import sqlite3InitModule from './sqlite3.mjs';

const steps = [];
const step = (name, value) => steps.push({ name, value });
const ms = (t0) => Math.round((performance.now() - t0) * 10) / 10;

async function run({ story }) {
  let t0 = performance.now();
  const sqlite3 = await sqlite3InitModule();
  step('init', { version: sqlite3.version.libVersion, ms: ms(t0) });

  t0 = performance.now();
  const pool = await sqlite3.installOpfsSAHPoolVfs({ name: 'qq-spike', initialCapacity: 6 });
  await pool.reserveMinimumCapacity(32);
  step('installOpfsSAHPoolVfs', { capacity: pool.getCapacity(), files: pool.getFileNames(), ms: ms(t0) });

  // Persistence: one row per run, so the count grows across reloads.
  const db = new pool.OpfsSAHPoolDb('/qq/spike.db');
  db.exec('CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, at TEXT NOT NULL)');
  db.exec({ sql: 'INSERT INTO runs (at) VALUES (?)', bind: [new Date().toISOString()] });
  const runs = db.selectValue('SELECT count(*) FROM runs');
  step('persistence', { runs, first: db.selectValue('SELECT min(at) FROM runs') });

  // BLOB round trip: a Float32Array vector in, the same bytes out.
  const vector = new Float32Array([0.25, -1.5, 3.125]);
  db.exec('CREATE TABLE IF NOT EXISTS blobs (id INTEGER PRIMARY KEY, v BLOB NOT NULL)');
  db.exec({ sql: 'INSERT INTO blobs (v) VALUES (?)', bind: [new Uint8Array(vector.buffer)] });
  const back = db.selectValue('SELECT v FROM blobs ORDER BY id DESC LIMIT 1');
  step('blob', {
    type: Object.prototype.toString.call(back),
    equal: Array.from(new Float32Array(back.buffer, back.byteOffset, back.byteLength / 4)).join() === Array.from(vector).join(),
  });

  // Rollback: a failed transaction leaves nothing behind.
  const before = db.selectValue('SELECT count(*) FROM blobs');
  try {
    db.exec('BEGIN');
    db.exec({ sql: 'INSERT INTO blobs (v) VALUES (?)', bind: [new Uint8Array(4)] });
    db.exec('INSERT INTO missing_table VALUES (1)');
    db.exec('COMMIT');
  } catch {
    db.exec('ROLLBACK');
  }
  step('rollback', { unchanged: db.selectValue('SELECT count(*) FROM blobs') === before });

  // sqlite-vec is not in this build: the core falls back to brute force (ADR 26).
  try {
    step('vec_version', db.selectValue('SELECT vec_version()'));
  } catch (error) {
    step('vec_version', `absent: ${String(error.message ?? error).slice(0, 80)}`);
  }
  db.close();

  // FileStore operations the core needs: copy = export + import, move = copy + unlink, size.
  t0 = performance.now();
  const bytes = pool.exportFile('/qq/spike.db');
  pool.importDb('/qq/saves/copy.db', bytes);
  const copy = new pool.OpfsSAHPoolDb('/qq/saves/copy.db');
  const copiedRuns = copy.selectValue('SELECT count(*) FROM runs');
  copy.close();
  const unlinked = pool.unlink('/qq/saves/copy.db');
  step('copy/unlink', { size: bytes.byteLength, copiedRuns, unlinked, files: pool.getFileNames(), ms: ms(t0) });

  // The real story: import 1.3 MB, read meta, brute-force search over its vectors.
  if (story) {
    t0 = performance.now();
    pool.importDb('/qq/stories/brass-vault.qqstory', new Uint8Array(story));
    const importMs = ms(t0);
    const s = new pool.OpfsSAHPoolDb('/qq/stories/brass-vault.qqstory');
    const tables = s.selectValues("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name");
    const vectorTable = tables.find((name) => /vector|embed/i.test(name));
    let search = null;
    if (vectorTable) {
      const cols = s.selectValues(`SELECT name FROM pragma_table_info('${vectorTable}')`);
      const blobCol = cols.find((c) => /vec|embedding/i.test(c));
      t0 = performance.now();
      const rows = s.selectValues(`SELECT ${blobCol} FROM ${vectorTable}`);
      const loadMs = ms(t0);
      const toF32 = (b) => new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
      const vectors = rows.map(toF32);
      const query = vectors[0];
      t0 = performance.now();
      let best = -Infinity;
      for (const v of vectors) {
        let dot = 0;
        for (let i = 0; i < v.length; i++) dot += v[i] * query[i];
        if (dot > best) best = dot;
      }
      search = { table: vectorTable, count: vectors.length, dims: query.length, loadMs, dotMs: ms(t0), best };
    }
    s.close();
    step('story', { importMs, tables, search, size: story.byteLength });
  }

  step('finalFiles', pool.getFileNames());
}

self.onmessage = async ({ data }) => {
  try {
    await run(data ?? {});
    self.postMessage({ ok: true, report: steps });
  } catch (error) {
    self.postMessage({ ok: false, report: steps, error: String(error?.stack ?? error) });
  }
};
