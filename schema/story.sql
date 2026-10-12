-- QuipQuill story file (.qqstory), schema version 1.
-- Single source of truth for the builder (Python) and the mobile core (TypeScript).
-- A story file is immutable once built; a playthrough copies it and adds its own tables.
-- Docs: docs/architecture/data-model.md

PRAGMA foreign_keys = ON;

-- Key/value metadata. Required keys (checked by `builder validate`):
--   schema_version, story_id, title, language, start_location,
--   world_summary, world_constraints, style, embed_model, embed_dim,
--   built_at, builder_version
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
) STRICT;

-- World entities. `id` is stable and prefixed by type: loc:, item:, npc:, org:, doc:, ent:.
CREATE TABLE entities (
  id          TEXT PRIMARY KEY,
  type        TEXT NOT NULL CHECK (type IN
                ('LOCATION', 'ITEM', 'NPC', 'CREATURE', 'ORGANIZATION', 'DOCUMENT', 'OTHER')),
  subtype     TEXT,                    -- extractor type: Spot, Building, Tool, Person, ...
  name        TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  props       TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(props))
) STRICT;

-- Directed relations. Normalized by the builder:
--   item/npc LOCATED_IN loc     (location HAS_ITEM / HAS_AGENT are rewritten to this form)
--   npc/player HAS_ITEM item    (inventory)
--   loc LEADS_TO loc            (reverse edge added unless the path is one-way)
--   HAS_ATTRIBUTE, INTERACTS_WITH kept as extracted
CREATE TABLE relations (
  id          INTEGER PRIMARY KEY,
  src         TEXT NOT NULL REFERENCES entities(id),
  dst         TEXT NOT NULL REFERENCES entities(id),
  type        TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '', -- for LEADS_TO: direction or path ("North", "through the oak door")
  UNIQUE (src, dst, type)
) STRICT;

CREATE INDEX relations_src ON relations (src, type);
CREATE INDEX relations_dst ON relations (dst, type);

-- Embeddings: float32 little-endian, length = meta.embed_dim.
--   kind 'name' — the entity name; kind 'desc' — "<name>: <description>" (only if described).
-- Queries at play time must use the same model (meta.embed_model).
CREATE TABLE embeddings (
  entity_id TEXT NOT NULL REFERENCES entities(id),
  kind      TEXT NOT NULL CHECK (kind IN ('name', 'desc')),
  vector    BLOB NOT NULL,
  PRIMARY KEY (entity_id, kind)
) STRICT;
