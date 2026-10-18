# Деплой облака: VPS, CI/CD

Облако работает на тестовом VPS и выкатывается из GitHub Actions. Решение — [ADR 40](../architecture/decisions.md#40-тестовый-сервер-vps--ghcr--github-actions).

| | |
|---|---|
| Адрес | `https://146-103-120-112.sslip.io` (WebSocket — `wss://…/ws`) |
| Сервер | Ubuntu 26.04, 1 vCPU, 1 GB RAM (+2 GB swap), 10 GB, IP `146.103.120.112` |
| Образ | `ghcr.io/bataevk/quipquill-cloud:<sha>` (и `:latest`) |
| Стек | [deploy/docker-compose.yml](../../deploy/docker-compose.yml): Postgres 17, RabbitMQ 4, migrate, cloud, Caddy (TLS от Let's Encrypt) |

`sslip.io` — бесплатный DNS: имя `146-103-120-112.sslip.io` указывает на этот IP, поэтому TLS работает без покупки домена. Свой домен: A-запись на IP, затем `DOMAIN` в `/opt/quipquill/.env` и переменная `DEPLOY_DOMAIN` окружения `production` в GitHub.

## Как работает CI/CD

- [`.github/workflows/cloud.yml`](../../.github/workflows/cloud.yml) — на каждый push и PR, затрагивающий `cloud/` или `deploy/`: ruff, pytest (Postgres и RabbitMQ — сервисы job'а). На `project-131027` (или кнопкой **Run workflow**) дальше: сборка образа → GHCR → окружение `production` → копирование `deploy/*` на сервер → `deploy.sh <образ>` → проверка `/health`, `/health/db`, `/health/mq` по HTTPS.
- [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) — builder (ruff, pytest) и мобильное ядро (`tsc`, `expo lint`, vitest).
- Окружение `production` разрешает деплой только из ветки `project-131027`. В нём секреты `DEPLOY_SSH_KEY` (ключ пользователя `deploy`), `DEPLOY_KNOWN_HOSTS`, **`CLOUD_SECRET_ENV`** и переменные `DEPLOY_HOST`, `DEPLOY_DOMAIN`, **`CLOUD_CONFIG_ENV`**.
- **Настройки облака — два env-файла в GitHub** (ADR 51): переменная `CLOUD_CONFIG_ENV` — несекретные (`DAILY_COST_LIMIT_USD`, `PROVIDER_ORDER`, `LOG_LEVEL`…), секрет `CLOUD_SECRET_ENV` — `JWT_SECRET`, `OPENROUTER_API_KEY`, `INVITE_CODES`. В каждую вставляется **файл целиком**, по шаблонам [deploy/cloud.config.env.example](../../deploy/cloud.config.env.example) и [deploy/cloud.secret.env.example](../../deploy/cloud.secret.env.example). Шаг **Env files** перед выкаткой пишет их в `/opt/quipquill/cloud.config.env` и `cloud.secret.env` (права 600); compose читает оба, секреты — последними. Пока значение в GitHub не задано, на сервере остаётся свой файл (старый `cloud.env` один раз становится `cloud.secret.env`), а в запуске — предупреждение.
- Образ тянется токеном самого job'а (`docker login` на время деплоя, затем `logout`), поэтому пакет в GHCR может оставаться приватным.

## Сервер

`/opt/quipquill` (владелец `deploy`):

| Файл | Что | Откуда |
|---|---|---|
| `docker-compose.yml`, `Caddyfile`, `deploy.sh` | стек | копирует CI из `deploy/` |
| `.env` | `DOMAIN`, `POSTGRES_PASSWORD`, `RABBITMQ_PASSWORD`, `CLOUD_IMAGE` (тег пишет `deploy.sh`) | создан вручную, не в git (переменные compose, не облака) |
| `cloud.config.env` | несекретные настройки облака | пишет CI из переменной `CLOUD_CONFIG_ENV` |
| `cloud.secret.env` | `JWT_SECRET`, `OPENROUTER_API_KEY`, `INVITE_CODES` | пишет CI из секрета `CLOUD_SECRET_ENV` |
| `cloud.env` | до ADR 51: всё вместе | больше не читается; удалить после переноса в GitHub |

Доступ: `ssh -i ~/.ssh/quipquill_vps root@146.103.120.112` (ключ администратора на Mac владельца). Вход по паролю отключён (`/etc/ssh/sshd_config.d/01-quipquill.conf`), открыты только 22, 80, 443 (ufw). Первичная настройка — [deploy/bootstrap.sh](../../deploy/bootstrap.sh) (повторный запуск безопасен).

Инвайт-код для регистрации — в `cloud.secret.env` (и в секрете `CLOUD_SECRET_ENV`):

```bash
ssh -i ~/.ssh/quipquill_vps root@146.103.120.112 grep INVITE /opt/quipquill/cloud.secret.env
```

## Частые действия

Статус и логи:

```bash
ssh -i ~/.ssh/quipquill_vps root@146.103.120.112 'cd /opt/quipquill && docker compose ps && docker compose logs --tail 50 cloud'
```

Откат на прошлый образ (теги — в GHCR или в истории запусков `cloud`):

```bash
ssh -i ~/.ssh/quipquill_vps root@146.103.120.112 'sudo -u deploy /opt/quipquill/deploy.sh ghcr.io/bataevk/quipquill-cloud:<sha>'
```

Если пакет приватный, перед ручным откатом нужен `docker login ghcr.io` с личным токеном (`read:packages`); проще перезапустить нужный старый запуск workflow (**Re-run jobs**).

Сменить настройку или секрет: отредактировать `CLOUD_CONFIG_ENV` / `CLOUD_SECRET_ENV` в окружении `production` (Settings → Environments → production) и перезапустить последний запуск `cloud` (**Re-run jobs**) или нажать **Run workflow**. Срочно, без CI: поправить файл на сервере и `docker compose up -d cloud` в `/opt/quipquill` — следующий деплой перезапишет его значением из GitHub.

Провайдер лежит: временно задать порядок для всех маршрутов — `PROVIDER_ORDER=["amazon-bedrock"]` в `CLOUD_CONFIG_ENV` (ADR 50; без переменной — порядок маршрута: Vertex global → Bedrock, ADR 51). Какой провайдер обслужил вызов и задержка — в логе `cloud`: строки `llm … provider= ttft= total=`.

## Приложение против сервера

```bash
EXPO_PUBLIC_CLOUD_URL=https://146-103-120-112.sslip.io npx expo start
```

## Перезапуск сервера

Docker и containerd включены в автозагрузку, у `postgres`, `rabbitmq`, `cloud`, `caddy` — `restart: unless-stopped`; `migrate` одноразовый и при перезапуске не нужен. Docker поднимает контейнеры без учёта `depends_on`, поэтому облако может стартовать раньше RabbitMQ: тогда оно подключается к очереди в фоне (2 → 30 с) и `/health/mq` до этого отвечает `down: connecting` (ADR 52). Проверка после перезагрузки:

```bash
for p in health health/db health/mq; do curl -s https://146-103-120-112.sslip.io/$p; echo; done
```

## Ограничения

- 1 GB RAM — впритык (Postgres, RabbitMQ и API вместе ~600–700 MB). При OOM — тариф 2 GB.
- Бэкапов Postgres нет: тестовый сервер. Перед реальными пользователями — `pg_dump` по расписанию ([backlog](../plan/backlog.md)).
- Деплой с коротким простоем: контейнер `cloud` пересоздаётся; клиент переподключается и делает `resume`, но генерации в памяти старого процесса теряются (ADR 18).
