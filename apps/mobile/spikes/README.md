# Spikes этапа 3.1 и проверки на устройстве

Проверочный код, на котором приняты ADR 25–27. Результаты — [docs/notes/spikes.md](../../../docs/notes/spikes.md).

| Файл | Что |
|---|---|
| `agent.ts` | LangChain `createAgent`: своя модель, инструмент, динамический промпт, Guard в `beforeAgent` |
| `sqlite-vec.ts` | op-sqlite + sqlite-vec: `vec0` из BLOB, KNN, перебор в JS для сравнения |
| `shims/` | `hermes-globals.ts` — полифилы spike (crypto здесь небезопасный — только для spike). Рабочие обходы с этапа 4.1 — в `src/platform/` (`polyfills.ts`, `shims/async-hooks.js`) |
| `screen.tsx` | экран, запускающий обе проверки на устройстве |
| `agent-check.tsx` | этап 4.1: мастер (`createAgent` + инструменты + op-sqlite) на Hermes по демо-сценарию со скриптовой моделью |
| `run-node.ts` | `node spikes/run-node.ts` — проверка агента в Node |

Обходы перенесены в `src/platform/` в этапе 4.1.

Веб (этап W.1): `web/` — SQLite WASM + OPFS и LangChain в браузере, см. [web/README.md](web/README.md).
