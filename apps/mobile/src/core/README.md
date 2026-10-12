# core

Ядро игры на чистом TypeScript — **без импортов** `react-native`, `expo-*` и op-sqlite. Платформа подключается через интерфейсы-порты, поэтому ядро тестируется в Node и позже переедет на десктоп.

Планируемые модули: `storage/` (порт БД), `world/` (репозиторий мира, события), `agent/` (мастер, middleware, инструменты), `net/` (WS-клиент, QueueChatModel), `protocol/` (типы сообщений). См. [agent](../../../../docs/architecture/agent.md) и [protocol](../../../../docs/architecture/protocol.md).
