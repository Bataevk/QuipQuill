// SQLite in the browser (stage W.1, ADR 69): the database runs in a Web Worker that the page
// loads from public/sqlite/, past Metro. This copies the pinned @sqlite.org/sqlite-wasm build
// there; the copies are not committed (.gitignore). Runs before `web` and `export:web`.
import { copyFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'node_modules/@sqlite.org/sqlite-wasm/dist');
const target = join(root, 'public/sqlite');

mkdirSync(target, { recursive: true });
// index.mjs finds the wasm next to itself (new URL('sqlite3.wasm', import.meta.url)).
copyFileSync(join(dist, 'index.mjs'), join(target, 'sqlite3.mjs'));
copyFileSync(join(dist, 'sqlite3.wasm'), join(target, 'sqlite3.wasm'));
console.log('sqlite-wasm → public/sqlite/');
