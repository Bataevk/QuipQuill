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

## 6.2f Живой прогон с общими инструментами
**Проведён 2026-10-17.** `QQ_RUNS=1 npm run test:live`: демо-сценарий 6.1 и «отказ от ограбления» на Haiku 5.5 — оба без ошибок, с инструментами `set_entity` / `set_relation` / `cancel_relation` вместо `move` / `take` / `add_entity` / … (ADR 43).

- Вещи персонажа на старте мастер создаёт `set_entity` и выдаёт `set_relation(player, has_item, …)` — 2–3 вещи, 6 событий; вступление $0,0019–0,0021 (было $0,0011), 3–4 вызова.
- Взять фонарь, подняться на Lamplight Row, выхватить визитку у Pip — `item.taken`, `player.moved`, `npc.reacted` + `item.taken`; ошибок аргументов нет.
- Отказ — `relation.set` мастером; генератор на том же ходу переписал описание Mags (`gen:entity.edited`) — пробел 6.2d в этот раз закрыт.
- Средний ход игрока — **$0,0027** (демо: $0,0020–0,0025, «выхватить» $0,0059; отказ: $0,0024–0,0031).

## 6.2d Живой прогон «отказ от ограбления»
**Проведён 2026-10-16.** `QQ_RUNS=0 npm run test:live` (`apps/mobile/src/core/game/demo.live.test.ts`, второй тест): Ada Finch, адаптация, выбор, затем 5 ходов — спросить Mags, отказаться («I'm out: I won't rob a bank for anyone»), попросить выпивку, уйти на Lamplight Row, спросить Pip, говорила ли о ней Mags. Haiku 5.5 через локальное облако, генератор после каждого хода игрока. До этого демо-сценарий 6.1 прошёл с генератором один раз без ошибок.

| Шаг | Время, с | Стоимость, $ | Вызовы | События (gen: — генератор) |
|---|---|---|---|---|
| Спросить Mags | 12,6 | 0,0019 | guard 1, gm 2, npc 1, generator 1 | npc.reacted, note.added |
| **Отказ** | 11,7 | 0,0021 | guard 1, gm 2, npc 1, generator 1 | npc.reacted, **relation.set** (мастер), note.added |
| Попросить выпивку | 10,2 | 0,0023 | guard 1, gm 2, npc 1, generator 1 | npc.reacted, **gen:relation.set** |
| Уйти на Lamplight Row | 12,1 | 0,0022 | guard 1, gm 2, generator 1 | player.moved |
| Спросить Pip | 10,5 | 0,0024 | guard 1, gm 2, npc 1, generator 1 | npc.reacted |
| **Средний ход игрока** | ~11,4 | **0,0022** | | |

- **Мир следует за игроком.** На отказе мастер сам записал связь Mags → игрок («Ada refused to join the Clockheart heist and walked away from the crew…») — правило промпта 6.2c работает. На следующем ходу генератор её уточнил («Mags has softened slightly toward Ada; she pours her a drink… though she still does not expect her to join the heist»).
- **NPC ведут себя по-новому.** Mags отвечает на отказ («every crew has a squall before the sailing… a locksmith with cold feet»), а на просьбу о выпивке наливает, но держит дистанцию. Pip уже знает: «Walked out on the job, did you? Heard the crew was grumblin' about that».
- **Не сделано:** описание Mags («plans the heist, and recruited the player») никто не переписал — ни мастер, ни генератор, хотя промпт генератора просит. Формально это прошлое и остаётся правдой, но может тянуть NPC к старому сюжету. Это вопрос для настройки промптов (6.2).
- **Стоимость.** Генератор добавляет ~$0,0003 за ход; средний ход ($0,0022) не дороже, чем в 6.1 ($0,0023), — сценарий короче, а кэш выше. Игрок генератор не ждёт: время ответа то же, ~10–12 с.
- **Попутно:** vitest 5 не показывает `console.log` прошедших тестов, поэтому `npm run test:live` теперь идёт с `--silent=false`, иначе таблицы замеров пропадают.

## 6.1 Сквозной прогон демо-сценария и стоимость хода
**Проведён 2026-10-16.** `npm run test:live` (`apps/mobile/src/core/game/demo.live.test.ts`): ядро в Node → локальное облако (`LLM_HANDLER=openrouter`, Bedrock первым) → `anthropic/claude-haiku-5.5`. Сценарий — `src/core/game/demo-script.ts`: новый персонаж (Ada Finch), адаптация, выбор, вступление и 7 ходов (осмотреться, спросить Mags, взять фонарь, уйти на Lamplight Row, спросить Pip, инвентарь, выхватить у Pip визитку).

**Результат: 3 прогона подряд без ошибок** (ни одного исключения, пустого ответа или обрыва по лимиту инструментов).

| Шаг | Время, с (среднее) | Стоимость, $ (среднее) |
|---|---|---|
| Адаптация персонажа | 4,4 | 0,0004 |
| Выбор + вступительная сцена | 9,4 | 0,0011 |
| Ход без NPC (взять, пойти, инвентарь) | 6–11 | 0,0017–0,0020 |
| Ход с реакцией NPC | 10–16 | 0,0020–0,0024 |
| Выхватить предмет у NPC (реакция + решение) | 19,4 | 0,0038 |
| **Средний ход игрока** (21 ход) | **12,1** | **0,0023** (максимум 0,0066) |
| **Прогон целиком** (9 шагов) | ~105 | **0,0177** (0,0169 / 0,0220 / 0,0141) |

- Вызовов на ход: Guard 1 + мастер 2–3 (+ NPC 1). Вход мастера растёт с историей: ~9 тыс. токенов на первых ходах, ~13 тыс. к седьмому; из кэша — около трети.
- Попытка выхватить визитку удалась в 1 прогоне из 3 (решает мастер) — так и задумано (ADR 32).
- Лимит free $0.05 в сутки ≈ 20 ходов или ~2,5 прогона демо — для показа инвестору задавать `DAILY_COST_LIMIT_USD`.

**Интерфейс:** строка стоимости под ходом и итог сессии — проверены на Android-эмуляторе (`$0.0019 · 3 calls (guard 1, gm 2) · 18.8K in (3.6K cached) · 487 out`). Реальный телефон — за владельцем (гайд по запуску).

**Попутно:** эмулятор Android на 4 ГБ RAM после долгого простоя уходил в подкачку, и система показывала «QuipQuill isn't responding» (ANR: `Input dispatching timed out`); после холодного старта эмулятора — без ANR. Приложение в debug-сборке занимает ~400 МБ (нативная куча ~240 МБ) уже на главном экране.

## 4.1 Мастер на Hermes
**Проведена 2026-10-15.** iPhone 17 simulator, iOS 27.0, dev-сборка (пересобрана: в старой не было `expo-secure-store`), `langchain` 1.5.16, `@langchain/core` 1.2.17, `@langchain/langgraph` 1.5.0, `langsmith` 0.10.13. Код — `apps/mobile/spikes/agent-check.tsx`.

Рабочий код ядра (`createGmAgent`, 9 инструментов, `WorldPrompt`, `InvalidToolCalls`, `ChatStore`, `runTurn`) на op-sqlite + sqlite-vec с обходами из `src/platform/` (а не из `spikes/shims`): демо-сценарий «осмотреться → пойти → взять → инвентарь» со скриптовой моделью.

**Результат:** `ok`; события `[] / player.moved / item.taken / []`, игрок на Lamplight Row с газетой, 14 сообщений в `model_messages`; 4 хода, 7 вызовов модели — **97 мс** (debug). Ошибок в логе Metro нет.

**Повтор после 4.2** (тот же день): `runTurn` на `agent.stream` (`for await`, история по шагам) со всеми middleware, кроме Guard и сводки (им нужна модель) — `ok`, **112 мс**.

## 3.1 Попутно: запуск на iOS 27
Сборка SDK 57 в Xcode 27 проходит, но приложение **падает при запуске** (`EXC_BREAKPOINT` в `_UIApplicationEvaluateRuntimeIssueForNoSceneLifecycleAdoption`): iOS 27 требует жизненный цикл UIScene у приложений, собранных с iOS 27 SDK, а шаблон SDK 57 его не использует. Исправлено бэкпортом Expo: `expo-build-properties` → `ios.enableSceneSupport: true` (Info.plist получает `UIApplicationSceneManifest` с `EXExpoAppSceneDelegate`). В SDK 58 флаг не нужен. Источники: [expo/expo#50179](https://github.com/expo/expo/issues/50179), [react-native-community/template#251](https://github.com/react-native-community/template/pull/251).

## 2.3 Кэширование промпта и effort через OpenRouter
**Проведена 2026-10-14** живыми запросами к `anthropic/claude-haiku-5.5` через OpenRouter (системный промпт ~2 600 токенов с `cache_control: ephemeral`).

- **Кэш:** на Bedrock второй вызов с тем же префиксом читает кэш (`cached_tokens = 2603`, стоимость $0.00003 вместо $0.00033). На Vertex global каждый вызов снова **пишет** кэш (+25%), чтение нестабильно. Без `cache_control` кэша нет ни у кого.
- **Порядок провайдеров:** без `order` OpenRouter выбирал Google, и кэш читался лишь на 4-м вызове; с `order: [amazon-bedrock, google-vertex/global]` — со 2-го.
- **Effort:** `reasoning.effort` `low` и `medium` принимаются; на простых запросах `reasoning_tokens = 0`.
- **Открытый риск:** в сквозном прогоне через Bedrock ~10 с до первого токена (через Vertex ~1 с) — замерить в 6.1.

**Решение:** Bedrock первым, точки кэширования ставит клиент — [ADR 20](../architecture/decisions.md#20-llm-шлюз-лимиты-effort-провайдеры-с-работающим-кэшем). Подробные факты — [research.md](research.md#openrouter-для-игровых-вызовов-2026-10-14).
