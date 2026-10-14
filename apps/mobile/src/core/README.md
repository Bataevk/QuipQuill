# core

Ядро игры на чистом TypeScript — **без импортов** `react-native`, `expo-*` и op-sqlite (проверяет ESLint). Платформа подключается через порты, поэтому ядро тестируется в Node и позже переедет на десктоп.

| Модуль | Что |
|---|---|
| `storage/` | Порты `Db` / `DbDriver` и `FileStore`, миграции по `PRAGMA user_version` |
| `story/` | Чтение и проверка meta файла `.qqstory` |
| `vector/` | Порт `VectorIndex`: sqlite-vec (`vec0`) и перебор в JS (ADR 26) |
| `library/` | `app.db` (персонажи, истории, сохранения), импорт истории, сохранения; фасад `Library` |
| `world/` | Мир партии: `WorldState` (запросы к `state_*`), `EventLog` (журнал, заметки), `World` (команды = события, ADR 29), имена и нечёткий поиск |
| `util/` | id (стандартный Web Crypto, ADR 27) |

Планируются: `net/` и `protocol/` (3.4), `agent/` (4.x). Адаптеры — в `src/platform/` (op-sqlite, expo-file-system) и `src/platform/node/` (node:sqlite, node:fs — тесты и десктоп). Тесты: `npm test`.
