# Spikes этапа 3.1

Проверочный код, на котором приняты ADR 25–27. Результаты — [docs/notes/spikes.md](../../../docs/notes/spikes.md).

| Файл | Что |
|---|---|
| `agent.ts` | LangChain `createAgent`: своя модель, инструмент, динамический промпт, Guard в `beforeAgent` |
| `sqlite-vec.ts` | op-sqlite + sqlite-vec: `vec0` из BLOB, KNN, перебор в JS для сравнения |
| `shims/` | обходы для Hermes: `async-hooks.js` (подключён в `metro.config.js`), `hermes-globals.ts` (полифилы; crypto здесь небезопасный — только для spike) |
| `screen.tsx` | экран, запускающий обе проверки на устройстве |
| `run-node.ts` | `node spikes/run-node.ts` — проверка агента в Node |

Перенос обходов в `src/platform/` — этап 4.1.
