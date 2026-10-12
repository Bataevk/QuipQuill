# QuipQuill mobile

Expo SDK 57 app (React Native 0.86, TypeScript, Expo Router). Native modules (op-sqlite) mean **Expo Go is not supported** — use a development build.

```bash
npm install
npx expo run:ios        # build the dev client and launch the simulator
npx expo start          # Metro only, once the dev client is installed
npm run typecheck
npm run lint
```

- `src/app/` — screens (Expo Router)
- `src/components/`, `src/hooks/`, `src/constants/` — UI building blocks
- `src/core/` — platform-independent game core (no React Native imports)

Rules for contributors and agents: [CLAUDE.md](CLAUDE.md), [AGENTS.md](AGENTS.md).
