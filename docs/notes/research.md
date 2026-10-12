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

**Проверить в этапе 2.3:** работает ли кэширование промпта через OpenRouter для Anthropic (`cache_control`, поле cache read в usage) и как передать effort.

## Мобильный стек (проверено 2026-10-12)

| Факт | Источник |
|---|---|
| Expo SDK 57 — стабильный (RN 0.86); SDK 58 — бета (RN 0.88 RC, iOS 27) | [SDK 57](https://expo.dev/changelog/sdk-57), [SDK 58 beta](https://expo.dev/changelog/sdk-58-beta) |
| op-sqlite поддерживает плагины FTS5, Rtree, cr-sqlite, **sqlite-vec**; включаются ключом в `package.json` (точное имя — в README) | [op-sqlite](https://docsearch.algolia.com/mcp/docs/repo/op-engineering/op-sqlite) |
| sqlite-vec на iOS и Android — **экспериментально** (предсобранные библиотеки с v0.1.2) | [релиз sqlite-vec](https://https-github_com.proxy.bugsmash.io/asg017/sqlite-vec/releases/tag/v0.1.2) |
| expo-sqlite тоже умеет sqlite-vec (`withSQLiteVecExtension` в плагине) — запасной путь | [expo-sqlite](https://docs.expo.dev/versions/latest/sdk/sqlite.md) |
| LangChain.js и LangGraph.js зависят от `AsyncLocalStorage` (`node:async_hooks`), которого нет в браузере и RN; в `@langchain/core` есть `MockAsyncLocalStorage`; хелперам нужно передавать `config` явно. **Подтверждений работы `createAgent` на Hermes не найдено** | [LangChain forum](https://forum.langchain.com/t/browser-compatibility-issue-createagent-fails-due-to-node-async-hooks-dependency/2713), [reference](https://reference.langchain.com/javascript/langchain-langgraph/web/getCurrentTaskInput) |
| LangChain JS v1: `createAgent` из `langchain`, middleware `beforeModel`, `wrapModelCall`, `afterModel`, `wrapToolCall`, `dynamicSystemPromptMiddleware` | [migration guide](https://docs.langchain.com/oss/javascript/migrate/langchain-v1) |
| `@langchain/langgraph-checkpoint-sqlite` рассчитан на Node — на RN нужен свой чекпоинтер или своя история в SQLite | [reference](https://reference.langchain.com/javascript/langchain-langgraph-checkpoint-sqlite) |
| Tauri 2 mobile работает, но по отзывам сырой (плагины, подпись, webview) | обзоры 2026 |

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
