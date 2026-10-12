# builder

Офлайн-сборщик историй: лор (текст) → граф (LLM-экстрактор) → `.qqstory` (SQLite + эмбеддинги + meta).

Стек: Python, uv. Экстрактор переносится из ветки `main` в этапе 1.1, экспорт — в этапе 1.2. См. [legacy-prototype](../docs/notes/legacy-prototype.md) и [data-model](../docs/architecture/data-model.md).
