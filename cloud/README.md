# cloud

Тонкий бэкенд QuipQuill: авторизация, шлюз к LLM и эмбеддингам (OpenRouter), очередь RabbitMQ + WebSocket. Игровой логики здесь нет. Протокол: [docs/architecture/protocol.md](../docs/architecture/protocol.md).

Стек: Python 3.13, uv, FastAPI, pydantic-settings, SQLAlchemy 2 + asyncpg + Alembic, PyJWT, pwdlib (argon2), PostgreSQL 17, RabbitMQ 4.

## Запуск (из корня репозитория)

```bash
make infra-up       # Postgres :5432 + RabbitMQ :5672 (UI http://localhost:15672, quipquill/quipquill)
make cloud-migrate  # применить миграции Alembic к базе из DATABASE_URL
make cloud-dev      # API с автоперезагрузкой: http://127.0.0.1:8000/health, документация /docs
make cloud-test     # тесты; БД-тесты используют базу quipquill_test (создаётся автоматически)
make cloud-lint
```

Новая миграция после изменения моделей: `cd cloud && uv run alembic revision --autogenerate -m "<что>"`, затем проверить файл в `migrations/versions/`.

## Авторизация

Email + пароль, регистрация только по инвайт-коду (`INVITE_CODES`).

| Эндпоинт | Что |
|---|---|
| `POST /auth/register` `{email, password, invite_code}` | 201 → `{access_token, refresh_token, token_type, expires_in}`; 403 `invalid_invite_code`; 409 `email_taken` |
| `POST /auth/login` `{email, password}` | 200 → пара токенов; 401 `invalid_credentials` |
| `POST /auth/refresh` `{refresh_token}` | 200 → новая пара; старый refresh отзывается. Повтор уже отозванного → отзыв **всех** токенов пользователя, 401 `refresh_token_reused` |
| `GET /me` (Bearer) | `{id, email, tier, created_at, usage_today}`; лимиты тарифа добавятся в 2.3 |

- Пароли — argon2id (`pwdlib`); email хранится в нижнем регистре.
- Access — JWT HS256 на 15 минут (`sub`, `tier`, `type=access`); ключ не короче 32 байт.
- Refresh — случайная строка на 30 дней; в БД только SHA-256.

Настройки читаются из переменных окружения и `cloud/.env` (шаблон — `.env.example`). Значения по умолчанию подходят для локальной инфраструктуры из `docker-compose.yml`. При `ENV=prod` сервис не стартует со стандартным `JWT_SECRET`.

## Структура `src/quipquill_cloud/`

| Пакет | Что | Этап |
|---|---|---|
| `main.py` | `create_app()`, подключение роутеров | 0.2 |
| `core/` | конфигурация (`Settings`), `security.py` — пароли, JWT, refresh-токены | 0.2, 2.1 |
| `api/` | HTTP-роуты: `health`, `auth`, `me`, `deps` (текущий пользователь); дальше `embeddings`, WS | 0.2+ |
| `db/` | модели SQLAlchemy 2 (async, asyncpg), сессии; миграции — `cloud/migrations/` | 2.1 |
| `mq/` | очереди RabbitMQ, диспетчер WS (не `queue`, чтобы не путать со stdlib) | 2.2 |
| `llm/` | воркер OpenRouter, эмбеддинги | 2.3 |
