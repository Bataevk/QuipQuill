# Технические проверки (spikes)

Результаты проверок рисков. Каждая запись: что проверяли, на чём (устройство, версии), результат, решение (со ссылкой на пункт в [decisions.md](../architecture/decisions.md)).

## 3.1a op-sqlite + sqlite-vec на устройстве
**Проведена 2026-10-15.** iPhone 17 simulator, iOS 27.0, Xcode 27.0, Expo SDK 57.0.28 / RN 0.86.3, Hermes, debug-сборка, MacBook Air M5. Код — `apps/mobile/spikes/`.

Сценарий как при импорте истории: 5 000 векторов × 1536 (float32, единичной длины) пишутся BLOB в обычную таблицу, из неё строится `vec0`, затем KNN-запрос рядом с вектором №42.

| Что | Результат |
|---|---|
| Версии | SQLite **3.53.4** (встроена в op-sqlite 18.2.5), sqlite-vec **v0.1.7-alpha.2** |
| Включение | `"op-sqlite": {"sqliteVec": true}` в `apps/mobile/package.json`, плагин Expo не нужен; расширение грузится при `open()` |
| `STRICT` + `CHECK (json_valid(...))` | работает: битый JSON отвергается |
| Запись 5 000 BLOB (`executeBatch`) | 52–104 мс |
| Построение `vec0` из BLOB | 0,86–0,92 с |
| KNN `MATCH … LIMIT 5` | **7,6 мс** на запрос, находит №42 |
| Перебор в JS: загрузка всех векторов | 20–28 мс (~30 МБ в памяти) |
| Перебор в JS: скалярное произведение | **209 мс** на запрос, находит №42 |

**Вывод:** sqlite-vec работает и в ~30 раз быстрее перебора; версия alpha, поэтому за портом с перебором в запасе — [ADR 26](../architecture/decisions.md#26-векторный-поиск-sqlite-vec-за-портом-перебор-в-js-запасной). Android не проверялся (нет Android Studio).

## 3.1b LangChain JS v1 `createAgent` на Hermes
**Проведена 2026-10-15.** iPhone 17 simulator, iOS 27.0, Xcode 27.0, Expo SDK 57.0.28 / RN 0.86.3, Hermes, debug-сборка, MacBook Air M5. Код — `apps/mobile/spikes/`. `langchain` 1.5.16, `@langchain/core` 1.2.17, `zod` 4.

Проверялось: своя модель-наследник `BaseChatModel` (прообраз `QueueChatModel`), инструмент `move` на zod, ошибка инструмента текстом, `dynamicSystemPromptMiddleware` (пересборка WORLD STATE перед каждым вызовом), свой `beforeAgent` с `jumpTo: 'end'` (Guard → reject без вызова модели).

**Результат:** все 6 проверок проходят и в Node, и на Hermes; ход из 3 вызовов модели — 22 мс в Node, 59 мс на симуляторе. Но без обходов LangChain на Hermes **не запускается**:

| # | Проблема | Обход (сейчас в `apps/mobile/spikes/shims` и конфигах) |
|---|---|---|
| 1 | Metro отвергает `import(config.package)` в `chat_models/universal.js` (`initChatModel`), его тянет `AgentNode` → ошибка сборки. Опция `dynamicDepsInPackages` действует только на `require` | Babel-плагин в `babel.config.js`: нелитеральный `import(x)` → `Promise.reject(...)`. Модель передаём объектом, ветка мёртвая |
| 2 | `node:async_hooks` (`@langchain/langgraph`, `@langchain/core`) | `metro.config.js` подменяет модуль на заглушку `AsyncLocalStorage` без контекста — как браузерные сборки LangChain. Следствие: неявная передача config (`getConfig()`, `interrupt()` без аргументов) не работает, config передаём явно |
| 3 | `crypto.getRandomValues` нет (uuid чекпоинтов LangGraph) | заглушка на `Math.random` **только для spike**; в приложении — стандартный Web Crypto через `expo-crypto` ([ADR 27](../architecture/decisions.md#27-случайные-числа-и-uuid-стандартный-web-crypto-на-rn--через-expo-crypto)) |
| 4 | `navigator.userAgent` нет → `isJsDom()` в `@langchain/core/utils/env` падает | `navigator.userAgent = 'ReactNative'` |
| 5 | `AbortSignal.prototype.throwIfAborted` нет — вызывается на каждом шаге стрима | полифил в 3 строки |

Есть в Hermes / RN 0.86 изначально: `AbortSignal.any`, `AbortSignal.timeout`, `structuredClone`, `ReadableStream`, `TextEncoder/Decoder`, `TextDecoderStream`.

**Цена:** production-бандл iOS (Hermes bytecode) **2,4 → 6,6 МБ** (+4,2 МБ) от `langchain` + `@langchain/langgraph` + `langsmith`.

**Решение владельца:** LangChain JS — [ADR 25](../architecture/decisions.md#25-агент-мастера-на-langchain-js-v1-с-обходами-для-hermes).

## 3.1 Попутно: запуск на iOS 27
Сборка SDK 57 в Xcode 27 проходит, но приложение **падает при запуске** (`EXC_BREAKPOINT` в `_UIApplicationEvaluateRuntimeIssueForNoSceneLifecycleAdoption`): iOS 27 требует жизненный цикл UIScene у приложений, собранных с iOS 27 SDK, а шаблон SDK 57 его не использует. Исправлено бэкпортом Expo: `expo-build-properties` → `ios.enableSceneSupport: true` (Info.plist получает `UIApplicationSceneManifest` с `EXExpoAppSceneDelegate`). В SDK 58 флаг не нужен. Источники: [expo/expo#50179](https://github.com/expo/expo/issues/50179), [react-native-community/template#251](https://github.com/react-native-community/template/pull/251).

## 2.3 Кэширование промпта и effort через OpenRouter
**Проведена 2026-10-14** живыми запросами к `anthropic/claude-haiku-5.5` через OpenRouter (системный промпт ~2 600 токенов с `cache_control: ephemeral`).

- **Кэш:** на Bedrock второй вызов с тем же префиксом читает кэш (`cached_tokens = 2603`, стоимость $0.00003 вместо $0.00033). На Vertex global каждый вызов снова **пишет** кэш (+25%), чтение нестабильно. Без `cache_control` кэша нет ни у кого.
- **Порядок провайдеров:** без `order` OpenRouter выбирал Google, и кэш читался лишь на 4-м вызове; с `order: [amazon-bedrock, google-vertex/global]` — со 2-го.
- **Effort:** `reasoning.effort` `low` и `medium` принимаются; на простых запросах `reasoning_tokens = 0`.
- **Открытый риск:** в сквозном прогоне через Bedrock ~10 с до первого токена (через Vertex ~1 с) — замерить в 6.1.

**Решение:** Bedrock первым, точки кэширования ставит клиент — [ADR 20](../architecture/decisions.md#20-llm-шлюз-лимиты-effort-провайдеры-с-работающим-кэшем). Подробные факты — [research.md](research.md#openrouter-для-игровых-вызовов-2026-10-14).
