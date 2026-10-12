# Старый прототип (StoryMind, ветка `main`)

Ветка `project-131027` создана почти с корня (`555dee2`), поэтому старого кода в рабочем дереве **нет**. Он лежит в `main` (HEAD `5a25926`, 2025). Как достать файл:

```bash
git show 5a25926:src/storymind/extractor.py
```

## Что было

Однопользовательская текстовая RPG на Python 3 + LangGraph 0.4 + Neo4j + ChromaDB + Flask.

| Файл в `main` | Что | Судьба |
|---|---|---|
| `src/storymind/extractor.py` (585 строк) | `GraphExtractor`: разбивка текста на куски → LLM извлекает сущности и связи (JSON) → склейка похожих имён по Левенштейну (порог 0.85) → сводка описаний | **Перенести** в `builder/` (этап 1.1) |
| `src/storymind/my_dataclasses.py` | `Entity`, `Relationship` | Перенести в builder |
| `scripts/utils.py` | `preprocess_text`, `remove_tokens`, `load_config`, `_strip_json_string` | Взять нужное в builder |
| `config.yaml` стр. 28–164 | `extraction_system_prompt` — промпт извлечения (типы связей: `LOCATED_IN, HAS_ITEM, HAS_AGENT, LEADS_TO, HAS_ATTRIBUTE, INTERACTS_WITH`) | Перенести в builder |
| `config.yaml` стр. 217–225 | Промпты сводки описаний | Перенести в builder |
| `config.yaml` стр. 359–416 | `system_lite` — промпт мастера | Основа для нового промпта (этап 4.1), но без перечисления инструментов |
| `config.yaml` стр. 227–358 | `system` — полная версия промпта мастера | Источник формулировок и примеров |
| `src/storymind/prompts.py` | Промпты валидатора (`passed/failed/edit`) и апдейтора | Идеи для Guard (4.2) и `add_entity` |
| `src/storymind/gm_tools.py` | 11 инструментов мастера | Набор → новые инструменты (4.1) |
| `src/storymind/manager.py` | Логика инструментов поверх двух графов + SQLite удалённых связей | **Не переносить** — заменено журналом событий |
| `src/storymind/agentgraph.py` | Граф LangGraph: trimmer → validator → updator → chatbot ⇄ tools | Не переносить — заменено одним агентом с middleware |
| `src/storymind/databeses/*` | Обёртки Neo4j (`GDBMod`), Chroma (`VDBMod`), `KnowledgeDB` с 4 видами поиска | Не переносить. Идея «поиск по именам и по описаниям» → две записи в `embeddings` (`kind = name / desc`) |
| `app.py`, `templates/index.html`, `cli.py` | Flask-веб-чат и CLI | Не переносить |
| `docs/Idea.md`, `docs/diagrams.md`, `docs/update_diagrams.md` | Исходная идея (ВКР), диаграммы | Справочно |
| `docs/test/Сравнение двух подходов.md` | Экстрактор против MiniRAG на Gemma 1B/4B: больше связей при меньшем входе | Справочно, аргумент за свой экстрактор |

## Известные проблемы старого кода (не повторять)

- `move_agent` **не проверял `LEADS_TO`** — можно было переместиться в любую локацию из базы.
- `restart()` выполнял `MATCH (n) DETACH DELETE n` по всей динамической базе, а `Manager` был один на процесс — несколько игр одновременно невозможны.
- Сущности различались по имени в нижнем регистре: коллизии, жёстко заданный `"player"`.
- Двойная запись «предмет в локации» (`LOCATED_IN` и `HAS_ITEM`) породила SQLite удалённых связей и сложную склейку в `_filter_duplicate_rels`.
- Валидатор: лишний вызов LLM на каждый ход, ответ разбирался поиском подстроки (`'passed' in result`).
- `describe_entity`: из-за приоритета условного выражения в `return` при отсутствии выходов (`LEADS_TO`) функция возвращала **пустую строку**.
- `requirements.txt` в UTF-16, содержит `pywin32` и сотни лишних пакетов (torch, LightRAG и др.).
- Состояние графа агента — `InMemorySaver`, после перезапуска терялось.

## Что сработало и стоит сохранить

- Принцип «состояние в графе, LLM рассказывает».
- Разделение лора (неизменного) и текущего состояния.
- Перевод ошибок инструментов в повествование («ты тянешься за парашютом, но хватаешь пустоту»).
- Защита от «я вижу у себя легендарный меч» — теперь это задача Guard.
- Экстрактор с короткими запросами хорошо работает даже на маленьких моделях.
