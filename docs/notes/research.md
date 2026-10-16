# Исследования и проверенные факты

Факты, на которых держатся решения. У каждого указаны дата проверки и источник. Перед тем как опираться на факт старше месяца, перепроверьте его.

## Модели и провайдеры (проверено 2026-10-12)

| Факт | Источник |
|---|---|
| `anthropic/claude-haiku-5.5` доступен на OpenRouter (вышел 2026-10-07), контекст 1M, инструменты, structured outputs, 5 провайдеров с автоматическим переключением | [OpenRouter](https://openrouter.ai/anthropic/claude-haiku-5.5) |
| Цена Haiku 5.5: $0.10 / $0.50 за 1M (вход / выход) при промпте ≤100K; **$0.50 / $2.50 при промпте >100K** | Справочник Claude API (2026-10-06); на OpenRouter сверить с его страницей цен |
| Haiku 5.5: адаптивный thinking включён по умолчанию; effort `low…max`, по умолчанию `medium`; `budget_tokens` → 400; отключить thinking можно только при effort ≤ `high` | Справочник Claude API |
| Haiku 5.5 поддерживает системные сообщения посреди диалога (не сбрасывают кэш) — проверить, пропускает ли их OpenRouter | Справочник Claude API; проверка — этап 2.3 |
| Для сравнения: Sonnet 5.5 — $2 / $10, Opus 5.5 — $4 / $20 за 1M | Справочник Claude API |
| OpenRouter отдаёт эмбеддинги: `POST /api/v1/embeddings`, например `openai/text-embedding-3-small`, `qwen/qwen3-embedding-0.6b` | [OpenRouter docs](https://openrouter.ai/docs/api-reference/embeddings) |

**Проверить в этапе 2.3:** работает ли кэширование промпта через OpenRouter для Anthropic (`cache_control`, поле cache read в usage) и как передать effort. — проверено 2026-10-14, см. «OpenRouter для игровых вызовов».

## OpenRouter: проверено вживую (2026-10-13)

| Факт | Как проверено |
|---|---|
| Haiku 5.5 sync обслуживают: google-vertex/global, claude-on-aws, azure/global, anthropic, amazon-bedrock, + региональные vertex/bedrock/azure (дороже на 10%). Все поддерживают `reasoning`, `reasoning_effort`, `tools`, `response_format` | `GET /api/v1/models/anthropic/claude-haiku-5.5/endpoints` |
| `anthropic/claude-haiku-5.5:batch` — $0.05 / $0.25 за 1M, обслуживает только провайдер `anthropic` | то же для `:batch` |
| Вызов с `provider.only = [amazon-bedrock, google-vertex/global]`, `allow_fallbacks: false`, `reasoning.effort = low` — работает, ответ за ~1 с через Google, `reasoning_tokens = 0`; в `usage` приходит `cost` | живой запрос |
| Batch API: `POST /api/v1/batches` → `202 validating`, затем `GET /api/v1/batches/:id` до `completed`; результаты — в самом объекте batch. В ответе `model` раскрывается в снапшот `anthropic/claude-haiku-5.5-20261007`. **Пробный batch из одного запроса был в статусе `in_progress` больше 25 минут** | живой запрос |
| Эмбеддинги на OpenRouter: 33 модели; `openai/text-embedding-3-small` — $0.02 за 1M (OpenAI, Azure); есть `:batch` | `GET /api/v1/embeddings/models` |
| `google/gemini-embedding-2` на Vertex доступен только со слагом `google-vertex` (регион выбирает OpenRouter); с `google-vertex/global` — 404. Поддерживает `dimensions` | живой запрос (вариант не выбран) |

## OpenRouter для игровых вызовов (2026-10-14)

| Факт | Как проверено |
|---|---|
| Стрим: OpenAI-формат; `tool_calls` приходят дельтами по `index` (`id` и имя — в первой, аргументы — кусками); `finish_reason` `tool_calls`/`stop`; `usage` с `cost` — в последнем чанке (с `stream_options.include_usage`) | сырые чанки |
| Кэш промпта (системный промпт ~2 600 токенов с `cache_control`): **Bedrock** — 2-й вызов `cached_tokens=2603`, $0.00003 вместо $0.00033; **Vertex global** — каждый вызов `cache_write` (+25%), чтение не гарантировано; **без `cache_control`** кэша нет нигде | живые запросы |
| `provider.only` = оба без `order` → OpenRouter выбрал Google, кэш прочитан лишь на 4-м вызове; с `order` Bedrock первым — со 2-го | живые запросы |
| `reasoning.effort` low/medium для Haiku 5.5 принимается; на простых запросах `reasoning_tokens = 0` | живые запросы |
| Эмбеддинги `openai/text-embedding-3-small`: 1536 измерений, `usage.cost` в ответе (≈ $0.00000012 за 2 строки) | живой запрос |
| Сквозной вызов из контейнера (`npc`, Bedrock первым): ~10 с до первого токена (5 heartbeat по 2 с), затем 18 дельт; прямые замеры через Vertex давали ~1 с — **проверить задержку в 6.1** | `make app-up` + WebSocket |

## Сборка демо-истории (2026-10-14)

| Факт | Как проверено |
|---|---|
| Batch Haiku 5.5 через OpenRouter: запрос из 1 сообщения — **больше 60 минут** без результата, из 13 — больше 37 минут, `completed: 0`. Сразу после отправки `GET /batches/:id` ~1 мин отдаёт 404 | живые batch 2026-10-13 |
| Sync-извлечение лора на 3 300 слов: 13 кусков, ~55 с, 0 ошибок; повторное извлечение даёт другие формулировки имён (нужны новые правки) | живой прогон |
| Расход ключа за этап 1.3 целиком (2 извлечения, 4 прогона эмбеддингов, 3 черновика meta, пробы): **$0.096** | `GET /api/v1/key` |

## Мобильный стек (проверено 2026-10-12)

| Факт | Источник |
|---|---|
| Expo SDK 57 — стабильный (RN 0.86); SDK 58 — бета (RN 0.88 RC, iOS 27) | [SDK 57](https://expo.dev/changelog/sdk-57), [SDK 58 beta](https://expo.dev/changelog/sdk-58-beta) |
| op-sqlite поддерживает плагины FTS5, Rtree, cr-sqlite, **sqlite-vec**; включаются ключом в `package.json` (точное имя — в README) | [op-sqlite](https://github.com/OP-Engineering/op-sqlite) |
| sqlite-vec на iOS и Android — **экспериментально** (предсобранные библиотеки с v0.1.2) | [релиз sqlite-vec](https://github.com/asg017/sqlite-vec/releases/tag/v0.1.2) |
| expo-sqlite тоже умеет sqlite-vec (`withSQLiteVecExtension` в плагине) — запасной путь | [expo-sqlite](https://docs.expo.dev/versions/latest/sdk/sqlite.md) |
| LangChain.js и LangGraph.js зависят от `AsyncLocalStorage` (`node:async_hooks`), которого нет в браузере и RN; в `@langchain/core` есть `MockAsyncLocalStorage`; хелперам нужно передавать `config` явно. **Подтверждений работы `createAgent` на Hermes не найдено** | [LangChain forum](https://forum.langchain.com/t/browser-compatibility-issue-createagent-fails-due-to-node-async-hooks-dependency/2713), [reference](https://reference.langchain.com/javascript/langchain-langgraph/web/getCurrentTaskInput) |
| LangChain JS v1: `createAgent` из `langchain`, middleware `beforeModel`, `wrapModelCall`, `afterModel`, `wrapToolCall`, `dynamicSystemPromptMiddleware` | [migration guide](https://docs.langchain.com/oss/javascript/migrate/langchain-v1) |
| `@langchain/langgraph-checkpoint-sqlite` рассчитан на Node — на RN нужен свой чекпоинтер или своя история в SQLite | [reference](https://reference.langchain.com/javascript/langchain-langgraph-checkpoint-sqlite) |
| Tauri 2 mobile работает, но по отзывам сырой (плагины, подпись, webview) | обзоры 2026 |
| op-sqlite 18.2.5: SQLite 3.53.4, sqlite-vec v0.1.7-alpha.2; конфиг `"op-sqlite"` ищется в ближайшем `package.json` вверх от `node_modules` | исходники пакета, spike 3.1a (2026-10-15) |
| iOS 27 убивает при запуске приложения, собранные с iOS 27 SDK без жизненного цикла UIScene; Expo SDK 57 ≥ 57.0.23 даёт бэкпорт `expo-build-properties` → `ios.enableSceneSupport` | [expo/expo#50179](https://github.com/expo/expo/issues/50179), проверено 2026-10-15 |
| Metro отвергает `import()` с нелитеральным аргументом всегда; `transformer.dynamicDepsInPackages: 'throwAtRuntime'` (по умолчанию) действует только на `require` | исходник `metro/src/ModuleGraph/worker/collectDependencies.js`, 2026-10-15 |
| Hermes / RN 0.86 без полифилов: нет `crypto.getRandomValues`, `navigator.userAgent`, `AbortSignal.prototype.throwIfAborted`; есть `AbortSignal.any/timeout`, `structuredClone`, `ReadableStream`, `TextDecoderStream` | spike 3.1b, 2026-10-15 |
| `expo-crypto` и `react-native-get-random-values` не заявляют поддержку Windows | поиск 2026-10-15 |

## Хранилища и инфраструктура

| Факт | Источник |
|---|---|
| Kuzu заархивирован 10.10.2025 (Apple купила Kùzu Inc.), форк — LadybugDB | [Julia Discourse](https://discourse.julialang.org/t/kuzudb-or-general-graphdbs/133139/5), [PyPI](https://pypi.org/project/kuzu/) |
| Расширения SQLite с Cypher: GraphQLite, sqlite-graph (альфа) | [GraphQLite](https://www.sourcepulse.org/projects/27495750) |
| PostgreSQL — лицензия PostgreSQL License (разрешительная) | postgresql.org/about/licence |
| RabbitMQ — MPL 2.0 | rabbitmq.com |
| Redis 8 — AGPLv3 / RSAL / SSPL; Valkey — BSD-3 | — |
| LiteLLM: бюджеты и лимиты на ключ или пользователя — в open source, SSO и аудит — платные. **24.03.2026 на PyPI были опубликованы вредоносные версии 1.82.7 и 1.82.8**, которые крали ключи | [docs](https://docs.litellm.ai/docs/proxy/users), [NetSPI](https://www.netspi.com/blog/executive-blog/ai-ml-pentesting/litellm-supply-chain-compromise/) |

## JEV (модель решений для NPC)

Описана в блогах Seele (2026): игра отфильтровывает недопустимые действия, модель выбирает одно, вызов — по событию или таймеру; результат проверяется, при сбое — запасной вариант. Есть второй продукт с тем же названием — Jev от TypeSafe (сентябрь 2026). Независимых бенчмарков нет. Источники: [Seele JEV NPC](https://www.seeles.ai/resources/blogs/jev-npc), [Towards AI](https://pub.towardsai.net/jev-doesnt-write-it-decides-games-today-company-data-with-care-computer-vision-next-21178d277475).

## Стоимость хода (2026-10-16, этап 6.1)

| Факт | Как проверено |
|---|---|
| Средний ход игрока — **$0.0023**, ~12 с; ход с реакцией NPC — до $0.0038 и ~19 с; вступление $0.0011, адаптация $0.0004; прогон демо из 9 шагов ≈ $0.018 | 3 живых прогона `npm run test:live`, [spikes 6.1](spikes.md#61-сквозной-прогон-демо-сценария-и-стоимость-хода) |
| Из кэша читается около трети входа мастера (Bedrock) | `usage.cacheRead` в `done` |

## WebSocket и таймеры после фона (2026-10-17, этап 6.2e)

| Факт | Источник |
|---|---|
| Пока приложение iOS приостановлено (suspended), оно не может работать с сетью; соединение умирает, а после возврата `onclose` / `onerror` могут не прийти вовсе — сокет выглядит открытым («зомби») | Apple DTS на форумах разработчиков ([716118](https://developer.apple.com/forums/thread/716118), [696310](https://developer.apple.com/forums/thread/696310)); практический разбор для RN ([AgileSoftLabs](https://www.agilesoftlabs.com/blog/2026/07/react-native-websocket-not-connecting)) |
| Мёртвый сокет надо распознавать самим: по `AppState` → `active` проверить связь (ping) или переподключиться, не полагаясь на `onclose` | Там же |
| Таймеры JS в фоне останавливаются и продолжают работу после возврата (пропущенные тики не догоняются); логику по времени сверять по часам при возврате | [Документация таймеров Vega (форк RN от Amazon)](https://developer.amazon.com/docs/react-native-vega/0.72/timers.html) — для обычного RN на iOS и Android официального описания не нашлось |
