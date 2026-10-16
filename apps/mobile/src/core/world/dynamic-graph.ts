import type { Db } from '../storage/db';
import type { Entity } from './model';

/**
 * Writes to the dynamic graph (`dyn_*`, ADR 42). The story's `entities` / `relations` are never
 * touched: a dynamic row overrides the story's, a `CANCEL` row hides one story relation, and the
 * `world_*` views put the two together. Call inside the command's transaction.
 */

/** Writes the whole entity: it replaces the story's entity with the same id, or is new. */
export async function putEntity(db: Db, seq: number, entity: Entity): Promise<void> {
  await db.run(
    `INSERT INTO dyn_entities (id, type, subtype, name, description, props, updated_seq)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT (id) DO UPDATE SET type = excluded.type, subtype = excluded.subtype,
       name = excluded.name, description = excluded.description, props = excluded.props,
       updated_seq = excluded.updated_seq`,
    [entity.id, entity.type, entity.subtype, entity.name, entity.description, JSON.stringify(entity.props), seq],
  );
}

/**
 * Makes `src -type-> dst` part of the world. A story relation that was cancelled comes back
 * (with the story's description unless `description` says otherwise); a relation the story does
 * not have is written to the dynamic graph. Without `description` an existing one is kept.
 */
export async function relate(
  db: Db,
  seq: number,
  src: string,
  dst: string,
  type: string,
  description?: string,
): Promise<void> {
  await db.run("DELETE FROM dyn_relations WHERE src = ? AND dst = ? AND type = 'CANCEL' AND description = ?", [
    src,
    dst,
    type,
  ]);
  const own = await dynamic(db, src, dst, type);
  if (own) {
    await db.run('UPDATE dyn_relations SET description = ?, updated_seq = ? WHERE id = ?', [
      description ?? own.description,
      seq,
      own.id,
    ]);
    return;
  }
  const story = await storyRelation(db, src, dst, type);
  if (story && (description === undefined || description === story.description)) return;
  await db.run(
    'INSERT INTO dyn_relations (id, src, dst, type, description, updated_seq) VALUES (?, ?, ?, ?, ?, ?)',
    [story?.id ?? (await nextRelationId(db)), src, dst, type, description ?? '', seq],
  );
}

/** Takes `src -type-> dst` out of the world: the dynamic row goes, a story relation is cancelled. */
export async function unrelate(db: Db, seq: number, src: string, dst: string, type: string): Promise<void> {
  await db.run("DELETE FROM dyn_relations WHERE src = ? AND dst = ? AND type = ? AND type <> 'CANCEL'", [
    src,
    dst,
    type,
  ]);
  const story = await storyRelation(db, src, dst, type);
  if (!story) return;
  await db.run(
    `INSERT INTO dyn_relations (id, src, dst, type, description, updated_seq) VALUES (?, ?, ?, 'CANCEL', ?, ?)
     ON CONFLICT DO NOTHING`,
    [story.id, src, dst, type, seq],
  );
}

/** Removes every placement of an item: `item LOCATED_IN x` and `x HAS_ITEM item`. */
export async function unplace(db: Db, seq: number, itemId: string): Promise<void> {
  const placements = await db.all<{ src: string; dst: string; type: string }>(
    `SELECT src, dst, type FROM world_relations
      WHERE (src = ? AND type = 'LOCATED_IN') OR (dst = ? AND type = 'HAS_ITEM')`,
    [itemId, itemId],
  );
  for (const r of placements) await unrelate(db, seq, r.src, r.dst, r.type);
}

/** Takes an entity out of wherever it is (`LOCATED_IN`), before putting it somewhere else. */
export async function unlocate(db: Db, seq: number, id: string): Promise<void> {
  const places = await db.all<{ dst: string }>(
    "SELECT dst FROM world_relations WHERE src = ? AND type = 'LOCATED_IN'",
    [id],
  );
  for (const r of places) await unrelate(db, seq, id, r.dst, 'LOCATED_IN');
}

type RelationRow = { id: number; description: string };

function dynamic(db: Db, src: string, dst: string, type: string): Promise<RelationRow | undefined> {
  return db.get<RelationRow>(
    "SELECT id, description FROM dyn_relations WHERE src = ? AND dst = ? AND type = ? AND type <> 'CANCEL'",
    [src, dst, type],
  );
}

function storyRelation(db: Db, src: string, dst: string, type: string): Promise<RelationRow | undefined> {
  return db.get<RelationRow>('SELECT id, description FROM relations WHERE src = ? AND dst = ? AND type = ?', [
    src,
    dst,
    type,
  ]);
}

/** Above every story and dynamic id, so new ties come after the story's in `ORDER BY id`. */
async function nextRelationId(db: Db): Promise<number> {
  const row = await db.get<{ id: number }>(
    `SELECT max(coalesce((SELECT max(id) FROM relations), 0),
                coalesce((SELECT max(id) FROM dyn_relations), 0)) + 1 AS id`,
  );
  return Number(row?.id ?? 1);
}
