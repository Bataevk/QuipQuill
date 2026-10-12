# builder

Офлайн-сборщик историй: лор (текст) → граф (LLM-экстрактор) → `.qqstory` (SQLite + эмбеддинги + meta; экспорт — этап 1.2).

Стек: Python 3.13, uv, `openai` (OpenAI-совместимый клиент к OpenRouter), `rapidfuzz`, `pydantic-settings`.

## Использование

```bash
cd builder
cp .env.example .env                 # вписать OPENROUTER_API_KEY
uv run builder extract ../stories/<id>/lore -o ../stories/<id>/graph.json
uv run pytest && uv run ruff check   # тесты без сети
```

Вход — папка с файлами `.md` и `.txt`. Параметры (модель, размер кусков, порог склейки, параллельность) задаются в `.env` или флагами, см. `uv run builder extract --help`.

## Как работает `extract`

1. Текст режется на куски по абзацам (~2000 символов, перекрытие 200).
2. Каждый кусок → LLM → JSON с сущностями и связями (промпт в `prompts.py`, перенесён из старого `config.yaml`). Не больше `MAX_CONCURRENCY` запросов одновременно; кусок с битым ответом пропускается и считается в `meta.failed_chunks`.
3. Похожие имена склеиваются (нормализация + `rapidfuzz.fuzz.ratio` ≥ 0.85). Имя сохраняется в исходном виде.
4. Если у сущности несколько описаний, они сводятся одним вызовом LLM.
5. Концы связей приводятся к каноническим именам; сущность, упомянутая только в связи, добавляется с типом `OTHER` и пустым описанием.

Формат `graph.json`: `{version: 1, meta, entities: [{name, type, description}], relationships: [{source, target, type, description}]}`. Связи ссылаются на сущности по `name`; стабильные ID появляются при экспорте (1.2).

Отличия от старого экстрактора (ветка `main`): см. [docs/notes/legacy-prototype.md](../docs/notes/legacy-prototype.md).
