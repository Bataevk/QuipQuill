# QuipQuill

Interactive fiction with an AI game master. A story world is a typed graph (locations, items, characters, paths); the game master is an LLM agent that changes the world only through tools and narrates the result. Text-only for now, co-op later.

> Status: 5-day prototype in progress. Plan: [docs/plan/roadmap.md](docs/plan/roadmap.md).

## Repository

| Path | What |
|---|---|
| [`apps/mobile`](apps/mobile) | Expo (React Native, TypeScript) app: UI, game master agent, local SQLite world |
| [`cloud`](cloud) | FastAPI: auth, LLM / embeddings gateway (OpenRouter), RabbitMQ + WebSocket |
| [`builder`](builder) | Python: lore text → story graph → `.qqstory` file |
| [`schema`](schema) | Shared `.qqstory` SQLite schema |
| [`stories`](stories) | Demo stories |
| [`docs`](docs) | Roadmap, architecture, notes (in Russian) |

## Getting started

Step-by-step run and play guide (in Russian): [docs/notes/run-guide.md](docs/notes/run-guide.md). Machine setup: [docs/notes/dev-setup.md](docs/notes/dev-setup.md). Short version for the app (macOS, Xcode with an iOS simulator installed):

```bash
cd apps/mobile
npm install
npx expo run:ios
```

## License

MIT, see [LICENSE](LICENSE).
