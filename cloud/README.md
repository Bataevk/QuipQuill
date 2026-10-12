# cloud

Тонкий бэкенд QuipQuill: авторизация, шлюз к LLM и эмбеддингам (OpenRouter), очередь RabbitMQ + WebSocket. Игровой логики здесь нет. Протокол: [docs/architecture/protocol.md](../docs/architecture/protocol.md).

Стек: Python 3.13, uv, FastAPI, pydantic-settings, PostgreSQL 17, RabbitMQ 4.

## Запуск (из корня репозитория)

```bash
make infra-up     # Postgres :5432 + RabbitMQ :5672 (UI http://localhost:15672, quipquill/quipquill)
make cloud-dev    # API с автоперезагрузкой: http://127.0.0.1:8000/health, документация /docs
make cloud-test
make cloud-lint
```

Настройки читаются из переменных окружения и `cloud/.env` (шаблон — `.env.example`). Значения по умолчанию подходят для локальной инфраструктуры из `docker-compose.yml`. При `ENV=prod` сервис не стартует со стандартным `JWT_SECRET`.

## Структура `src/quipquill_cloud/`

| Пакет | Что | Этап |
|---|---|---|
| `main.py` | `create_app()`, подключение роутеров | 0.2 |
| `core/` | конфигурация (`Settings`) | 0.2 |
| `api/` | HTTP-роуты: `health` (дальше `auth`, `embeddings`, WS) | 0.2+ |
| `db/` | SQLAlchemy, Alembic | 2.1 |
| `mq/` | очереди RabbitMQ, диспетчер WS (не `queue`, чтобы не путать со stdlib) | 2.2 |
| `llm/` | воркер OpenRouter, эмбеддинги | 2.3 |
