# Spike W.1: SQLite WASM + OPFS в Expo web

Результаты — [docs/notes/spikes.md](../../../../docs/notes/spikes.md#w1-sqlite-wasm--opfs-в-expo-web).

| Файл | Что |
|---|---|
| `opfs-worker.mjs` | module worker: `installOpfsSAHPoolVfs`, сохранность после перезагрузки, BLOB, откат, copy/unlink через `exportFile`/`importDb`, импорт `.qqstory` и перебор векторов |
| `screen.tsx` | экран проверки: worker + `runAgentSpike()` из `../agent.ts` (LangChain в браузере) |

Повторить:
1. `cp spikes/web/screen.tsx src/app/spike.tsx`, поправить импорты (`'../agent'` → `'../../spikes/agent'`), `cp spikes/web/opfs-worker.mjs public/sqlite/`.
2. `npm run web` (копирует sqlite-wasm в `public/sqlite/`) и открыть `http://localhost:8081/spike`. Обновить страницу: `runs` растёт на 1.
3. Браузеры, которыми агент не управляет (Safari, Firefox), шлют отчёт POST-запросом на `http://localhost:8092/`. Простейший приёмник — node-сервер, который дописывает тело запроса в файл и отвечает с `access-control-allow-origin: *`.
4. Удалить `src/app/spike.tsx` и `public/sqlite/opfs-worker.mjs`.
