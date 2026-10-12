# Модель данных

Три хранилища:

| Хранилище | Где | Что |
|---|---|---|
| `*.qqstory` | файл SQLite, собирается builder | Граф истории (неизменяемый) + эмбеддинги + meta |
| `saves/<id>.db` | телефон | Копия `.qqstory` + состояние партии, журнал событий, чаты |
| `app.db` | телефон | Персонажи, список историй и сохранений |
| PostgreSQL | облако | Пользователи, токены обновления, учёт использования |

Схема `.qqstory` — единственный источник: [`schema/story.sql`](../../schema/story.sql) (таблицы `STRICT`, проверки типов через `CHECK`). Ниже — сокращённое описание; при расхождении прав файл схемы. Как builder заполняет файл — [builder/README.md](../../builder/README.md).

## `.qqstory` (история)

```sql
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- Обязательные ключи:
--   schema_version   '1'
--   story_id         uuid
--   title            'The Rune Halls'
--   language         'en'
--   start_location   'loc:entrance-rune-hall'
--   world_summary    краткое описание мира для мастера (1–3 абзаца)
--   world_constraints ограничения мира: эпоха, допустимые расы, магия/технологии (для адаптации персонажа)
--   style            тон и жанр повествования (подставляется в {{STORY_STYLE}})
--   embed_model      'openai/text-embedding-3-small' (запросы при игре — той же моделью!)
--   embed_dim        '1536'
--   built_at, builder_version

CREATE TABLE entities (
  id          TEXT PRIMARY KEY,      -- 'loc:tavern', 'item:rusty-key', 'npc:innkeeper'
  type        TEXT NOT NULL,         -- LOCATION | ITEM | NPC | CREATURE | ORGANIZATION | DOCUMENT | OTHER
  subtype     TEXT,                  -- Spot | Building | Tool | Person ... (из экстрактора)
  name        TEXT NOT NULL,
  description TEXT NOT NULL,
  props       TEXT NOT NULL DEFAULT '{}'  -- JSON: количество, флаги, персона NPC и т.п.
);

CREATE TABLE relations (
  id          INTEGER PRIMARY KEY,
  src         TEXT NOT NULL REFERENCES entities(id),
  dst         TEXT NOT NULL REFERENCES entities(id),
  type        TEXT NOT NULL,         -- LOCATED_IN | HAS_ITEM | LEADS_TO | HAS_AGENT | HAS_ATTRIBUTE | INTERACTS_WITH
  description TEXT NOT NULL DEFAULT '',  -- для LEADS_TO: направление/путь ("north, through the oak door")
  UNIQUE (src, dst, type)
);
CREATE INDEX relations_src ON relations(src, type);
CREATE INDEX relations_dst ON relations(dst, type);

CREATE TABLE embeddings (
  entity_id TEXT NOT NULL REFERENCES entities(id),
  kind      TEXT NOT NULL,           -- 'name' | 'desc'
  vector    BLOB NOT NULL,           -- float32 little-endian, длина = embed_dim
  PRIMARY KEY (entity_id, kind)
);
```

### Соглашения о связях и ID

- ID: `loc:`, `item:`, `npc:` (NPC и существа), `org:`, `doc:`, `ent:` + slug имени без ведущего артикля.
- Предмет или NPC в локации: `X → LOCATED_IN → loc`. Формы `loc → HAS_ITEM → item` и `loc → HAS_AGENT → npc` builder переписывает в `LOCATED_IN`, чтобы не было двух способов записать одно и то же (в старом прототипе из-за двойной записи появился костыль с удалёнными связями). `HAS_AGENT` в файле истории не встречается.
- Инвентарь: `npc|player → HAS_ITEM → item`.
- Проход: `loc → LEADS_TO → loc`. Builder добавляет обратное ребро (с пустым описанием), если в описании прохода нет «one-way».
- Эмбеддинги: `name` — имя, `desc` — «имя: описание» (только у описанных сущностей).
- Дополнительный ключ meta `source_extract_model` — модель, которой извлекался граф.

## Сохранение партии (`saves/<id>.db`)

Создаётся копированием `.qqstory`. Исходные `entities` и `relations` **не меняются** и нужны для перезапуска и сравнения. При старте добавляются таблицы:

```sql
CREATE TABLE state_entities  ( ...как entities... , updated_seq INTEGER NOT NULL );
CREATE TABLE state_relations ( ...как relations..., updated_seq INTEGER NOT NULL );
-- Заполняются копией entities/relations при session.started.
-- Все запросы мира во время игры идут в state_*.

CREATE TABLE events (
  seq     INTEGER PRIMARY KEY AUTOINCREMENT,
  turn    INTEGER NOT NULL,
  actor   TEXT NOT NULL,             -- 'player:<id>' | 'gm' | 'npc:<id>' | 'system'
  kind    TEXT NOT NULL,             -- см. ниже
  payload TEXT NOT NULL,             -- JSON
  ts      TEXT NOT NULL              -- ISO-8601
);

CREATE TABLE notes (                 -- заметки мастера: квесты, обещания, сюжетные нити
  id INTEGER PRIMARY KEY, text TEXT NOT NULL, created_seq INTEGER NOT NULL, closed_seq INTEGER
);

CREATE TABLE model_messages (        -- рабочая память моделей; сжимается
  chat_id TEXT NOT NULL,             -- 'gm' | 'npc:<id>'
  seq     INTEGER NOT NULL,
  role    TEXT NOT NULL,             -- system | user | assistant | tool
  content TEXT NOT NULL,             -- JSON сообщения (LangChain-совместимый)
  PRIMARY KEY (chat_id, seq)
);

CREATE TABLE transcript (            -- то, что видел игрок; только дописывается
  seq     INTEGER PRIMARY KEY AUTOINCREMENT,
  speaker TEXT NOT NULL,             -- 'player' | 'gm' | 'npc:<id>' | 'system'
  text    TEXT NOT NULL,
  extra   TEXT NOT NULL DEFAULT '{}',-- JSON: карточки выбора и т.п.
  ts      TEXT NOT NULL
);

CREATE VIRTUAL TABLE vec_entities USING vec0(...);  -- строится при импорте, если sqlite-vec доступен (решение 3.1)
```

### События

Одно событие = одна транзакция: `INSERT INTO events` + изменение `state_*`.

| `kind` | `payload` | Изменение состояния |
|---|---|---|
| `session.started` | `{story_id, character_id}` | копирование в `state_*`, сущность игрока |
| `character.adapted` | `{original, adapted, choice}` | обновление сущности игрока |
| `player.moved` | `{from, to}` | `LOCATED_IN` игрока |
| `item.taken` | `{item, from}` | `LOCATED_IN` → `HAS_ITEM` |
| `item.dropped` | `{item, to}` | `HAS_ITEM` → `LOCATED_IN` |
| `entity.added` | `{entity, relation}` | новая сущность + связь |
| `entity.edited` | `{id, before, after}` | описание или props |
| `player.updated` | `{before, after}` | описание игрока |
| `note.added` / `note.closed` | `{id, text}` | таблица `notes` |
| `npc.asked` | `{npc, question, answer}` | только журнал |

Мастер получает текущее состояние из `state_*` и **последние 5 событий** (улучшение отбора — в backlog).

## `app.db` (приложение)

```sql
CREATE TABLE characters (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, sex TEXT, age INTEGER, race TEXT,
  description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE TABLE stories ( id TEXT PRIMARY KEY, title TEXT, path TEXT, imported_at TEXT );
CREATE TABLE saves   ( id TEXT PRIMARY KEY, story_id TEXT, character_id TEXT, path TEXT,
                       created_at TEXT, updated_at TEXT, title TEXT );
```

## PostgreSQL (cloud)

```sql
users          (id uuid PK, email citext UNIQUE, password_hash text, tier text DEFAULT 'free',
                created_at timestamptz)
refresh_tokens (id uuid PK, user_id uuid FK, token_hash text, expires_at timestamptz,
                revoked_at timestamptz NULL)
usage_events   (id bigserial PK, user_id uuid FK, request_id text, session_id text,
                purpose text, model text, input_tokens int, output_tokens int,
                cache_read_tokens int, cost_usd numeric(12,6), created_at timestamptz)
-- лимиты: агрегат usage_events за сутки по user_id против конфигурации тарифа
```

Миграции — Alembic (этап 2.1).
