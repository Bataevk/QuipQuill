# QuipQuill

Интерактивная книга с ИИ-мастером: мир истории — граф (локации, предметы, NPC), мастер — LLM-агент, меняющий мир только через инструменты. Стартап, сейчас идёт **прототип на 5 дней** (ветка `project-131027`). Основной рынок англоязычный.

## Где что

- План и статусы этапов: [docs/plan/roadmap.md](docs/plan/roadmap.md) — **начинать отсюда**. Отложенное: [docs/plan/backlog.md](docs/plan/backlog.md).
- Архитектура: [overview](docs/architecture/overview.md) · [decisions (ADR)](docs/architecture/decisions.md) · [data-model](docs/architecture/data-model.md) · [protocol](docs/architecture/protocol.md) · [agent](docs/architecture/agent.md).
- Заметки: [legacy-prototype](docs/notes/legacy-prototype.md) (старый Python-код в `main`, `5a25926`) · [ai-samba-ideas](docs/notes/ai-samba-ideas.md) · [research](docs/notes/research.md) · [spikes](docs/notes/spikes.md) · [dev-setup](docs/notes/dev-setup.md) · [run-guide](docs/notes/run-guide.md) (запуск и ручная проверка) · [deploy](docs/notes/deploy.md) (тестовый сервер, CI/CD).

| Папка | Стек |
|---|---|
| `apps/mobile` | Expo SDK 57, React Native 0.86, TypeScript, op-sqlite. См. `apps/mobile/AGENTS.md` |
| `apps/mobile/src/core` | Ядро на чистом TS **без импортов React Native**: агент, мир, хранилище (порт + адаптеры), сеть, протокол. Тесты — в Node |
| `cloud/` | Python 3.13, uv, FastAPI, PostgreSQL, RabbitMQ (aio-pika), OpenRouter |
| `builder/` | Python, uv: лор → граф → `.qqstory` (SQLite + эмбеддинги) |
| `schema/story.sql` | Единая схема `.qqstory` для Python и TS |

## Как работать с этапами

1. Задача вида «выполни этап X.Y» → прочитать этап в roadmap и документы из строки «Читать».
2. Делать только то, что входит в этап. Замеченное за его рамками — одной строкой в `backlog.md` («Замечено во время этапов»).
3. Проверить по пункту «Готово, когда». Не объявлять этап готовым без зелёных тестов, линтера и проверки типов (`tsc` для TypeScript; для Python — тесты и ruff, ADR 23).
4. Обновить статус в таблице roadmap. Если решение изменилось — новый пункт в `decisions.md` (старые не переписывать), факты — в `research.md`, итоги проверок — в `spikes.md`.

## Правила

- **Промпты для LLM — на английском.** Документация и общение — на русском.
- Мир меняется только через инструменты → событие в `events` + обновление `state_*` в одной транзакции. Мастер не узнаёт о мире ничего, кроме промпта и результатов инструментов.
- Модель, лимиты и `max_tokens` выбирает облако по `purpose` и тарифу, а не клиент.
- Держать контекст хода **ниже 100K токенов** (выше этого Haiku 5.5 в 5 раз дороже). Стабильная часть промпта — в начале (кэш).
- Ошибки инструментов возвращать текстом, а не исключением. Каждый внешний вызов — с явным таймаутом.
- Из AI Samba (проект работодателя) переносятся только идеи, **не код**.
- Модель прототипа — `anthropic/claude-haiku-5.5` через OpenRouter. Не подставлять другие модели без запроса.
- Секреты — только в `.env` (не коммитить); пример — `.env.example`.

## Команды

```bash
# apps/mobile (запускать из apps/mobile, в корне нет package.json)
npx expo install <pkg>      # вместо npm install <pkg>
npx expo run:ios            # dev-сборка (Expo Go не подходит: op-sqlite нативный)
npx tsc --noEmit && npx expo lint && npm test   # npm test — ядро в Node (vitest)
npm run test:cloud          # ядро ↔ локальное облако (после make infra-up)

# cloud (из корня; make help — все цели)
make infra-up               # Postgres + RabbitMQ
make cloud-dev              # FastAPI на :8000
make cloud-test && make cloud-lint
make cloud-test-live        # живые вызовы OpenRouter (доли цента)
make app-up                 # весь бэкенд в Docker на :8080

# builder (из builder/)
uv run builder extract <папка_с_лором> -o graph.json
uv run builder export graph.json -o story.qqstory --start "<локация>"
uv run builder meta story.qqstory && uv run builder validate story.qqstory
uv run pytest && uv run ruff check
```

Известные ловушки окружения: см. раздел «Важно» в [dev-setup](docs/notes/dev-setup.md) (запуск из корня, полноширинный `＠`, `npm audit fix --force`).
