/**
 * Tables a save adds to its copy of the story file (data-model.md#сохранение-партии).
 * The story's own `entities` / `relations` / `embeddings` — the static graph — never change.
 * What the playthrough changes lives in the dynamic graph `dyn_*`; the game reads both through
 * the `world_*` views, the dynamic graph first (ADR 42). Append-only, tracked in user_version.
 */
export const SAVE_SCHEMA_STEPS = [
  // v1 (stage 3.3): a full copy of the story in state_* — replaced by v2.
  `CREATE TABLE state_entities (
     id          TEXT PRIMARY KEY,
     type        TEXT NOT NULL,
     subtype     TEXT,
     name        TEXT NOT NULL,
     description TEXT NOT NULL DEFAULT '',
     props       TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(props)),
     updated_seq INTEGER NOT NULL
   ) STRICT;
   CREATE TABLE state_relations (
     id          INTEGER PRIMARY KEY,
     src         TEXT NOT NULL REFERENCES state_entities (id),
     dst         TEXT NOT NULL REFERENCES state_entities (id),
     type        TEXT NOT NULL,
     description TEXT NOT NULL DEFAULT '',
     updated_seq INTEGER NOT NULL,
     UNIQUE (src, dst, type)
   ) STRICT;
   CREATE INDEX state_relations_src ON state_relations (src, type);
   CREATE INDEX state_relations_dst ON state_relations (dst, type);
   CREATE TABLE events (
     seq     INTEGER PRIMARY KEY AUTOINCREMENT,
     turn    INTEGER NOT NULL,
     actor   TEXT NOT NULL,
     kind    TEXT NOT NULL,
     payload TEXT NOT NULL CHECK (json_valid(payload)),
     ts      TEXT NOT NULL
   ) STRICT;
   CREATE TABLE notes (
     id          INTEGER PRIMARY KEY,
     text        TEXT NOT NULL,
     created_seq INTEGER NOT NULL,
     closed_seq  INTEGER
   ) STRICT;
   CREATE TABLE model_messages (
     chat_id TEXT NOT NULL,
     seq     INTEGER NOT NULL,
     role    TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
     content TEXT NOT NULL CHECK (json_valid(content)),
     PRIMARY KEY (chat_id, seq)
   ) STRICT;
   CREATE TABLE transcript (
     seq     INTEGER PRIMARY KEY AUTOINCREMENT,
     speaker TEXT NOT NULL,
     text    TEXT NOT NULL,
     extra   TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(extra)),
     ts      TEXT NOT NULL
   ) STRICT;`,
  // v2 (stage 6.2a, ADR 42): the dynamic graph over the story. A v1 save that has started
  // cannot be carried over (SaveManager refuses it before this step runs).
  `DROP TABLE state_relations;
   DROP TABLE state_entities;
   -- A row replaces the story's entity with the same id as a whole; new entities live only here.
   CREATE TABLE dyn_entities (
     id          TEXT PRIMARY KEY,
     type        TEXT NOT NULL,
     subtype     TEXT,
     name        TEXT NOT NULL,
     description TEXT NOT NULL DEFAULT '',
     props       TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(props)),
     updated_seq INTEGER NOT NULL
   ) STRICT;
   -- A relation replaces the story's one with the same (src, dst, type). type = 'CANCEL' hides
   -- one story relation (src, dst, <description>). id: the story relation's id when it replaces
   -- or cancels one, above every story id otherwise — the order of ties stays the story's.
   CREATE TABLE dyn_relations (
     id          INTEGER PRIMARY KEY,
     src         TEXT NOT NULL,
     dst         TEXT NOT NULL,
     type        TEXT NOT NULL,
     description TEXT NOT NULL DEFAULT '',
     updated_seq INTEGER NOT NULL,
     CHECK (type <> 'CANCEL' OR description <> '')
   ) STRICT;
   CREATE UNIQUE INDEX dyn_relations_key
     ON dyn_relations (src, dst, type, CASE WHEN type = 'CANCEL' THEN description ELSE '' END);
   CREATE INDEX dyn_relations_dst ON dyn_relations (dst, type);
   CREATE VIEW world_entities AS
     SELECT id, type, subtype, name, description, props FROM dyn_entities
     UNION ALL
     SELECT id, type, subtype, name, description, props FROM entities
      WHERE id NOT IN (SELECT id FROM dyn_entities);
   CREATE VIEW world_relations AS
     SELECT id, src, dst, type, description FROM dyn_relations WHERE type <> 'CANCEL'
     UNION ALL
     SELECT r.id, r.src, r.dst, r.type, r.description FROM relations r
      WHERE NOT EXISTS (
        SELECT 1 FROM dyn_relations d
         WHERE d.src = r.src AND d.dst = r.dst
           AND (d.type = r.type OR (d.type = 'CANCEL' AND d.description = r.type)));`,
  // v3 (stage 6.2b): vectors of new and rewritten entities. A row is valid only while `text` is
  // what the entity's name / "name: description" is now; a stale one is skipped and recomputed.
  `CREATE TABLE dyn_embeddings (
     entity_id TEXT NOT NULL,
     kind      TEXT NOT NULL CHECK (kind IN ('name', 'desc')),
     text      TEXT NOT NULL,
     vector    BLOB NOT NULL,
     PRIMARY KEY (entity_id, kind)
   ) STRICT;`,
  // v4 (stage 6.2g, ADR 43): actors instead of NPC chats. A note with `entity_id` is that
  // entity's actor note; reactions become `entity.reacted`; the NPC chats are no longer read.
  `ALTER TABLE notes ADD COLUMN entity_id TEXT;
   CREATE INDEX notes_entity ON notes (entity_id);
   UPDATE events
      SET kind = 'entity.reacted',
          payload = json_object('entity', json_extract(payload, '$.npc'),
                                'situation', json_extract(payload, '$.situation'),
                                'reaction', json_extract(payload, '$.reaction'))
    WHERE kind = 'npc.reacted';
   DELETE FROM model_messages WHERE chat_id LIKE 'npc:%';`,
  // v5 (stage 6.2k, ADR 48): which player turns already had their background work (generator,
  // entity memory), so a game reopened after the app was killed can finish it. Turns played
  // before count as done.
  `CREATE TABLE after_turns (
     turn        INTEGER PRIMARY KEY,
     finished_at TEXT NOT NULL
   ) STRICT;
   INSERT INTO after_turns (turn, finished_at)
     SELECT DISTINCT turn, '' FROM events WHERE turn > 0;`,
] as const;

/** The schema version whose saves can no longer be played once started (state_* copies). */
export const OUTDATED_SAVE_VERSION = 1;
