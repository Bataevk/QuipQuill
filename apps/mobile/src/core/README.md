# core

Ядро игры на чистом TypeScript — **без импортов** `react-native`, `expo-*` и op-sqlite (проверяет ESLint). Платформа подключается через порты, поэтому ядро тестируется в Node и позже переедет на десктоп.

| Модуль | Что |
|---|---|
| `storage/` | Порты `Db` / `DbDriver` и `FileStore`, миграции по `PRAGMA user_version` |
| `story/` | Чтение и проверка meta файла `.qqstory` |
| `vector/` | Порт `VectorIndex`: sqlite-vec (`vec0`) и перебор в JS (ADR 26) |
| `library/` | `app.db` (персонажи, истории, сохранения), импорт истории, сохранения; фасад `Library` |
| `util/` | id (стандартный Web Crypto, ADR 27) |

Планируются: `world/` (3.3), `net/` и `protocol/` (3.4), `agent/` (4.x). Адаптеры — в `src/platform/` (op-sqlite, expo-file-system) и `src/platform/node/` (node:sqlite, node:fs — тесты и десктоп). Тесты: `npm test`.
