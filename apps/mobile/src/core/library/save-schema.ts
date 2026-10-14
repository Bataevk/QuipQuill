/**
 * Tables a save adds to its copy of the story file (data-model.md#сохранение-партии).
 * The story's own `entities` / `relations` stay untouched; the game reads and writes `state_*`,
 * which `session.started` fills from them (stage 3.3). Append-only, tracked in user_version.
 */
export const SAVE_SCHEMA_STEPS = [
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
] as const;
