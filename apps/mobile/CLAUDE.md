@AGENTS.md

## QuipQuill: правила мобильного приложения

- `src/core/` — чистый TypeScript без импортов `react-native`, `expo-*` и op-sqlite. Платформенные вещи (БД, файлы, secure store, сеть) — через интерфейсы-порты, адаптеры лежат вне `core` (например, `src/platform/`). Это нужно, чтобы ядро тестировалось в Node и переехало на десктоп.
- Экраны — в `src/app/` (Expo Router), компоненты — в `src/components/`.
- Протокол с облаком — строго по [docs/architecture/protocol.md](../../docs/architecture/protocol.md); `done.message` — источник истины для агента, дельты — только для UI.
- Перед тем как объявить задачу готовой: `npx tsc --noEmit`, `npx expo lint`, `npm test` (тесты ядра, vitest).
- Тесты ядра — рядом с кодом (`src/core/**/*.test.ts`), окружение — `src/platform/node/test-env.ts` (node:sqlite + sqlite-vec, фикстура — демо-история).
