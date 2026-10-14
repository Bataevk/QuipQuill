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
  props       TEXT NOT NULL DEFAULT '{}'  -- JSON: {"fixed": true} — нельзя взять (ADR 19); позже количество и т.п.
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
- Предмет может лежать **в другом предмете** (`brass punch card LOCATED_IN wall safe`, ключ за котлом) — инструменты мастера должны это учитывать (4.1).
- Локация, которая `LOCATED_IN` достижимой локации, считается её частью (ADR 19).

## Сохранение партии (`saves/<id>.db`)

Создаётся копированием `.qqstory`. Исходные `entities` и `relations` **не меняются** и нужны для перезапуска и сравнения. При старте добавляются таблицы:

```sql
-- Точная схема — apps/mobile/src/core/library/save-schema.ts (PRAGMA user_version сохранения).
CREATE TABLE state_entities  ( ...как entities, без CHECK на type (появится игрок)..., updated_seq INTEGER NOT NULL );
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

CREATE VIRTUAL TABLE vec_entities USING vec0(embedding float[<embed_dim>]);
-- строится при импорте в копии истории; rowid = rowid строки embeddings; без расширения — перебор в JS (ADR 26)
```

### События

Одно событие = одна транзакция: `INSERT INTO events` + изменение `state_*`.

| `kind` | `payload` | Изменение состояния |
|---|---|---|
| `session.started` | `{story_id, character_id, player}` | копирование в `state_*`, сущность игрока `player:<character_id>` (тип `PLAYER`, анкета — в `props.original`) `LOCATED_IN` стартовой локации |
| `character.adapted` | `{original, adapted, choice}` | имя, описание, `props.race` игрока |
| `player.moved` | `{from, to}` | `LOCATED_IN` игрока; только по `LEADS_TO` из текущей локации |
| `item.taken` | `{item, from}` | все размещения предмета (`item LOCATED_IN x`, `x HAS_ITEM item`) → `player HAS_ITEM item`; предмет `ITEM`/`DOCUMENT`, не `fixed`, лежит здесь или внутри того, что лежит здесь |
| `item.dropped` | `{item, to}` | `HAS_ITEM` → `LOCATED_IN` текущей локации |
| `entity.added` | `{entity, relation: {type, target}}` | новая сущность (`ITEM`/`DOCUMENT`/`NPC`/`CREATURE`/`OTHER`, имя уникально, `props.added`) + `LOCATED_IN` текущей локации или `player HAS_ITEM` |
| `entity.edited` | `{id, before, after}` | описание и/или props (слияние) |
| `player.updated` | `{before, after}` | описание игрока |
| `note.added` / `note.closed` | `{id, text}` | таблица `notes` |
| `npc.asked` | `{npc, question, answer}` | только журнал |

Код: [`src/core/world/`](../../apps/mobile/src/core/world/) — `WorldState` (запросы), `EventLog` (журнал, заметки), `World` (команды). Нарушенное правило — не исключение, а результат `{ok: false, error}` с текстом для модели (ADR 29); транзакция при этом откатывается. Каждое изменённое `state_*` получает `updated_seq` своего события.

Мастер получает текущее состояние из `state_*` и **последние 5 событий** (улучшение отбора — в backlog).

## `app.db` (приложение)

```sql
CREATE TABLE characters (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, sex TEXT, age INTEGER, race TEXT,
  description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
) STRICT;
CREATE TABLE stories (                -- одна запись на story_id; новая сборка заменяет старую
  id TEXT PRIMARY KEY,                -- meta.story_id (стабилен между сборками, ADR 24)
  title TEXT NOT NULL, path TEXT NOT NULL, built_at TEXT NOT NULL DEFAULT '',
  vector_index TEXT NOT NULL,         -- 'sqlite-vec' | 'brute-force' (ADR 26)
  imported_at TEXT NOT NULL
) STRICT;
CREATE TABLE saves (
  id TEXT PRIMARY KEY, story_id TEXT NOT NULL REFERENCES stories(id),
  character_id TEXT REFERENCES characters(id), title TEXT NOT NULL, path TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
) STRICT;
```

- Схема — в коде: [`src/core/library/app-db.ts`](../../apps/mobile/src/core/library/app-db.ts); версии — `PRAGMA user_version`, шаги миграций только дописываются.
- Файлы лежат в каталоге данных приложения (`Documents` на iOS): `app.db`, `stories/<story_id>.qqstory`, `saves/<save_id>.db`. **Пути в таблицах — относительные**: на iOS путь контейнера приложения меняется между установками и обновлениями.
- Импорт: проверка `schema_version` и обязательных ключей meta → копия в `stories/` (оригинал не меняется) → `vec_entities` в копии → запись в `stories`. Повторный импорт той же истории заменяет файл; сохранения не страдают — у каждого своя копия.

## PostgreSQL (cloud)

Модели — [`cloud/src/quipquill_cloud/db/models.py`](../../cloud/src/quipquill_cloud/db/models.py), миграции — `cloud/migrations/` (Alembic).

```sql
users          (id uuid PK, email varchar(320) UNIQUE  -- хранится в нижнем регистре
                , password_hash text                    -- argon2id
                , tier varchar(16) DEFAULT 'free' CHECK (tier IN ('free','pro','max'))
                , created_at timestamptz)
refresh_tokens (id uuid PK, user_id uuid FK → users ON DELETE CASCADE (index),
                token_hash varchar(64) UNIQUE           -- SHA-256, сам токен не хранится
                , expires_at timestamptz, revoked_at timestamptz NULL, created_at timestamptz)
usage_events   (id bigserial PK, user_id uuid FK → users ON DELETE CASCADE, request_id, session_id,
                purpose, model, input_tokens, output_tokens, cache_read_tokens,
                cost_usd numeric(12,6), created_at timestamptz;
                INDEX (user_id, created_at))
-- лимиты (2.3): сумма usage_events за UTC-сутки по user_id против конфигурации тарифа
```
