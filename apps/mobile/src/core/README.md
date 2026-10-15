# core

Ядро игры на чистом TypeScript — **без импортов** `react-native`, `expo-*` и op-sqlite (проверяет ESLint). Платформа подключается через порты, поэтому ядро тестируется в Node и позже переедет на десктоп.

| Модуль | Что |
|---|---|
| `storage/` | Порты `Db` / `DbDriver` и `FileStore`, миграции по `PRAGMA user_version` |
| `story/` | Чтение и проверка meta файла `.qqstory` |
| `vector/` | Порт `VectorIndex`: sqlite-vec (`vec0`) и перебор в JS (ADR 26) |
| `library/` | `app.db` (персонажи, истории, сохранения), импорт истории, сохранения; фасад `Library` |
| `world/` | Мир партии: `WorldState` (запросы к `state_*`), `EventLog` (журнал, заметки), `World` (команды = события, ADR 29), имена и нечёткий поиск |
| `protocol/` | Типы протокола v1 (`messages.ts`, синхронно с `cloud/…/mq/schemas.py`), `StreamText` — текст генерации для UI, `LiveFeed` — живые пузыри экрана игры (поздние повторы `done` после `resume` не показываются) |
| `net/` | `AuthSession` (токены, refresh), `CloudSocket` (WS: auth, переподключение, resume, ping), `QueueChatModel` (LangChain-модель поверх очереди), `CloudEmbeddings` |
| `agent/` | Мастер (agent.md): промпт и WORLD STATE, инструменты над `World`, `createAgent` + `middleware/` (BugGuard, Guard, Context, ToolLimit, InvalidToolCalls, WorldPrompt), история в `model_messages` (`ChatStore`, `ChatSync`), `transcript`, `runTurn()` и `retryTurn()` с блокировкой хода (`turn-lock.ts`), NPC (`npc.ts`: промпт, чат, `npc_react`), старт сессии и адаптация персонажа (`character.ts`, `session.ts`), модели поверх очереди (`models.ts`); `testing/` — модель по сценарию и демо-сценарий |
| `game/` | `Game` — открытая партия для интерфейса (5.2): сохранение, мир, мастер и NPC поверх очереди, шаги игры (`start`, `clarify`, `choose`, `send`, `recover`), фаза, `nextStep()` (что нужно, чтобы партия не застряла), поток событий своей сессии |
| `util/` | id (стандартный Web Crypto, ADR 27) |
 Адаптеры — в `src/platform/` (op-sqlite, expo-file-system) и `src/platform/node/` (node:sqlite, node:fs — тесты и десктоп). Тесты: `npm test`; против локального облака — `npm run test:cloud`.
