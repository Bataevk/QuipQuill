# builder

Офлайн-сборщик историй: лор (текст) → граф (LLM-экстрактор) → `.qqstory` (SQLite по схеме [`schema/story.sql`](../schema/story.sql) + эмбеддинги + meta).

Стек: Python 3.13, uv, `openai` (OpenAI-совместимый клиент к OpenRouter), `rapidfuzz`, `pydantic-settings`.

## Использование

```bash
cd builder
cp .env.example .env                 # вписать OPENROUTER_API_KEY

# 1. лор → граф
uv run builder extract ../stories/<id>/lore -o ../stories/<id>/graph.json
# 2. граф → файл истории (эмбеддинги считаются здесь)
uv run builder export ../stories/<id>/graph.json -o ../stories/<id>/<id>.qqstory \
    --title "The Rune Halls" --start "The Entrance Rune Hall"
# 3. черновик world_summary / world_constraints / style одним вызовом LLM, затем ручная правка
uv run builder meta ../stories/<id>/<id>.qqstory
uv run builder meta ../stories/<id>/<id>.qqstory --set style="Dark fantasy, terse second person."
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

Формат `graph.json`: `{version: 1, meta, entities: [{name, type, description}], relationships: [{source, target, type, description}]}`. Связи ссылаются на сущности по `name`; стабильные ID появляются при экспорте.

## Как работает `export`

1. **Типы.** Тип экстрактора → тип истории: Spot/Building/Country → `LOCATION`, Person → `NPC`, Creature → `CREATURE`, Tool/Vehicle → `ITEM`, Document → `DOCUMENT`, Organization → `ORGANIZATION`, остальное → `OTHER`. Исходный тип сохраняется в `subtype`. Безтиповая сущность на конце `LEADS_TO` становится `LOCATION`.
2. **ID.** `<префикс>:<slug>`: `loc:`, `item:`, `npc:` (NPC и существа), `org:`, `doc:`, `ent:`. Ведущие артикли отбрасываются (`The Entrance Rune Hall` → `loc:entrance-rune-hall`), при совпадении добавляется `-2`, `-3`.
3. **Связи** приводятся к одной форме: `локация HAS_ITEM / HAS_AGENT X` → `X LOCATED_IN локация`; `HAS_ITEM` у NPC остаётся (инвентарь); для каждого `LEADS_TO` добавляется обратный проход, если в описании нет «one-way».
4. **Проверка структуры** до эмбеддингов (не платим за битый граф). Ошибки: нет стартовой локации, локации недостижимы от старта по `LEADS_TO`, висячие связи. Предупреждения: пустые описания, предмет нигде не лежит, предмет в нескольких местах, неизвестный тип связи.
5. **Эмбеддинги** через OpenRouter батчами по 64: `name` — имя, `desc` — «имя: описание» (только если описание есть). float32 little-endian.
6. Файл пишется во временный и переименовывается только если ошибок нет.

`builder validate` без `--draft` требует ещё и заполненных `world_summary`, `world_constraints`, `style`.

Отличия от старого экстрактора (ветка `main`): см. [docs/notes/legacy-prototype.md](../docs/notes/legacy-prototype.md).
