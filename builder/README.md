# builder

Офлайн-сборщик историй: лор (текст) → граф (LLM-экстрактор) → `.qqstory` (SQLite по схеме [`schema/story.sql`](../schema/story.sql) + эмбеддинги + meta).

Стек: Python 3.13, uv, `openai` (OpenAI-совместимый клиент к OpenRouter), `httpx2` (Batch API), `rapidfuzz`, `pydantic-settings`.

## Модели и провайдеры

- Чат: `anthropic/claude-haiku-5.5`, `reasoning.effort = low`, `temperature` не передаётся.
- **Batch по умолчанию** для `extract` (куски и сводки описаний) и для эмбеддингов в `export`: [Batch API OpenRouter](https://openrouter.ai/docs/projects/docs/batch-quickstart), −50% цены, результат через минуты–часы (окно 24 ч). Batch-вызовы Haiku идут на endpoint Anthropic `:batch`, провайдер не указывается. Флаг `--sync` переключает на обычные вызовы.
- Обычные (sync) вызовы чата идут **только** через `amazon-bedrock` и `google-vertex/global` с `allow_fallbacks: false` — эти провайдеры не сохраняют промпты (`PROVIDER_ONLY`). `builder meta` — всегда sync.
- Эмбеддинги: `openai/text-embedding-3-small`, 1536 измерений, без ограничения провайдера.
- Если batch не уложился в `BATCH_TIMEOUT`, задача продолжает выполняться на стороне OpenRouter; повторите позже или запустите с `--sync`.

## Использование

```bash
cd builder
cp .env.example .env                 # вписать OPENROUTER_API_KEY

# 1. лор → граф
uv run builder extract ../stories/<id>/lore -o ../stories/<id>/graph.json
# 1.5. ручные правки графа (воспроизводимо, см. ниже)
uv run builder curate ../stories/<id>/graph.json ../stories/<id>/curation.toml -o ../stories/<id>/graph.curated.json
# 2. граф → файл истории (эмбеддинги считаются здесь)
#    --meta: story_id, title, language и тексты мира берутся из meta.toml (при первом запуске он создаётся)
uv run builder export ../stories/<id>/graph.json -o ../stories/<id>/<id>.qqstory \
    --title "The Rune Halls" --start "The Entrance Rune Hall" --meta ../stories/<id>/meta.toml
# 3. черновик world_summary / world_constraints / style одним вызовом LLM, затем ручная правка;
#    с --meta правки попадают и в meta.toml, поэтому переживают пересборку
uv run builder meta ../stories/<id>/<id>.qqstory --meta ../stories/<id>/meta.toml
uv run builder meta ../stories/<id>/<id>.qqstory --meta ../stories/<id>/meta.toml --set style="Dark fantasy, terse second person."
uv run builder meta ../stories/<id>/<id>.qqstory --show
# 4. финальная проверка (в CI и перед коммитом истории)
uv run builder validate ../stories/<id>/<id>.qqstory

uv run pytest && uv run ruff check   # тесты без сети
```

Вход — папка с файлами `.md` и `.txt`. Параметры (модель, размер кусков, порог склейки, параллельность) задаются в `.env` или флагами, см. `uv run builder extract --help`.

## Как работает `extract`

1. Текст режется на куски по абзацам (~2000 символов, перекрытие 200).
2. Каждый кусок → LLM → JSON с сущностями и связями (промпт в `prompts.py`, перенесён из старого `config.yaml`). Не больше `MAX_CONCURRENCY` запросов одновременно; кусок с битым ответом пропускается и считается в `meta.failed_chunks`.
3. Похожие имена склеиваются (нормализация + `rapidfuzz.fuzz.ratio` ≥ 0.85). Имя сохраняется в исходном виде.
4. Если у сущности несколько описаний, они сводятся одним вызовом LLM.
5. Концы связей приводятся к каноническим именам; сущность, упомянутая только в связи, добавляется с типом `OTHER` и пустым описанием.

Персонажи и существа (Person, Creature) извлекаются и склеиваются **без сжатия** — отдельным промптом (`PERSONA_MERGE_SYSTEM`): знания, тайны и манера речи NPC нужны мастеру и `ask`.

## Ручные правки: `curate`

Файл `stories/<id>/curation.toml` (формат — docstring `curation.py`): `[merge]` дубликатов, `[drop]` мусора, `[retype]`, `[describe]`, `[props]` (например `{ fixed = true }` — нельзя взять), `[[drop_relation]]`, `[[add_relation]]`. Строгий режим: правило, не нашедшее сущность или связь, — ошибка (так видно, что после повторного извлечения имена поменялись). Пример — [stories/brass-vault/curation.toml](../stories/brass-vault/curation.toml).

Формат `graph.json`: `{version: 1, meta, entities: [{name, type, description}], relationships: [{source, target, type, description}]}`. Связи ссылаются на сущности по `name`; стабильные ID появляются при экспорте.

## Как работает `export`

1. **Типы.** Тип экстрактора → тип истории: Spot/Building/Country → `LOCATION`, Person → `NPC`, Creature → `CREATURE`, Tool/Vehicle → `ITEM`, Document → `DOCUMENT`, Organization → `ORGANIZATION`, остальное → `OTHER`. Исходный тип сохраняется в `subtype`. Безтиповая сущность на конце `LEADS_TO` становится `LOCATION`.
2. **ID.** `<префикс>:<slug>`: `loc:`, `item:`, `npc:` (NPC и существа), `org:`, `doc:`, `ent:`. Ведущие артикли отбрасываются (`The Entrance Rune Hall` → `loc:entrance-rune-hall`), при совпадении добавляется `-2`, `-3`.
3. **Связи** приводятся к одной форме: `локация HAS_ITEM / HAS_AGENT X` → `X LOCATED_IN локация`; `HAS_ITEM` у NPC остаётся (инвентарь); для каждого `LEADS_TO` добавляется обратный проход, если в описании нет «one-way».
3a. **Места внутри мест.** Локация `LOCATED_IN` другой локации без своих `LEADS_TO` (и без вложенных мест) получает двусторонний проход к родителю: «into …» / «back out to …» (ADR 31). Здание, которое только группирует комнаты с проходами, не меняется.
4. **Проверка структуры** до эмбеддингов (не платим за битый граф). Ошибки: нет стартовой локации, локации недостижимы от старта по `LEADS_TO`, висячие связи. Предупреждения: пустые описания, предмет нигде не лежит, предмет в нескольких местах, неизвестный тип связи.
5. **Эмбеддинги** через OpenRouter батчами по 64: `name` — имя, `desc` — «имя: описание» (только если описание есть). float32 little-endian.
6. Файл пишется во временный и переименовывается только если ошибок нет.

`builder validate` без `--draft` требует ещё и заполненных `world_summary`, `world_constraints`, `style`.

### `meta.toml` (ADR 24)

`stories/<id>/meta.toml` хранит то, что должно пережить пересборку: `story_id` (на него ссылаются сохранения на телефоне — импорт узнаёт по нему новую версию той же истории), `title`, `language` и авторские тексты `world_summary`, `world_constraints`, `style`. Без `--meta` каждый `export` создаёт новый `story_id` и пустые тексты. Если у истории уже есть `.qqstory`, а `meta.toml` нет, — `builder meta <story> --meta <toml> --show` создаст его из файла истории. `--title` при существующем `meta.toml` игнорируется (меняйте `title` в файле).

Отличия от старого экстрактора (ветка `main`): см. [docs/notes/legacy-prototype.md](../docs/notes/legacy-prototype.md).
