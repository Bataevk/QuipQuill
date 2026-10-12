# Окружение разработки (macOS)

## Состояние машины разработчика (2026-10-13)

| Инструмент | Состояние |
|---|---|
| Homebrew, watchman, fnm, uv | ✅ (uv также есть в `~/.local/bin`, эта версия идёт в PATH первой) |
| Node | ✅ v26.11.1, npm 11.20 |
| Xcode | ✅ 27.0, `xcode-select` указывает на Xcode.app |
| iOS Simulator runtime | ❌ **не установлен** — Xcode → Settings → Components → iOS |
| Android Studio | ❌ не установлен (нужен для Android-сборок) |
| Docker Desktop | ✅ 29.5 |
| `apps/mobile` | ✅ Expo SDK 57.0.28, RN 0.86.3, `expo-dev-client`, `@op-engineering/op-sqlite` 18.2 |

## Важно

- Все команды `npx expo …` и `npm …` для приложения запускаются **из `apps/mobile`**. В корне репозитория нет `package.json`.
- **Expo Go не подходит**: op-sqlite — нативный модуль. Нужна dev-сборка: `npx expo run:ios` (локально) или `eas build --profile development` (в облаке).
- Пакеты ставить через `npx expo install <пакет>` — он подбирает версии, совместимые с SDK. **Не запускать `npm audit fix --force`**: сломает согласованность версий. Чинить версии — `npx expo install --fix`.
- Следите за символом `@` в именах пакетов: полноширинный `＠` (U+FF20) уже ломал установку.
- Expo быстро меняется — сверяться с документацией своей версии: https://docs.expo.dev/versions/v57.0.0/ (см. `apps/mobile/AGENTS.md`).

## Команды

```bash
# мобильное приложение (из apps/mobile)
npx expo run:ios            # dev-сборка + запуск в симуляторе
npx expo start              # только Metro, если dev-сборка уже установлена
npx tsc --noEmit            # проверка типов
npx expo lint               # линтер
npx expo-doctor             # диагностика зависимостей

# бэкенд (после этапа 0.2, из корня)
docker compose up -d        # Postgres + RabbitMQ
cd cloud && uv run fastapi dev

# builder (после этапа 1.1)
cd builder && uv run builder --help
```

Отладка: React Native DevTools — клавиша `j` в терминале Metro; меню разработчика — `Cmd+D` в iOS-симуляторе; RabbitMQ UI — http://localhost:15672.

## Расширения VS Code

| Расширение | Зачем |
|---|---|
| Expo Tools (`expo.vscode-expo-tools`) | Автодополнение `app.json` / `eas.json`, отладка |
| React Native Tools (`msjsdiag.vscode-react-native`) | Отладчик |
| ESLint, Prettier, Error Lens | Качество кода |
| Python, Pylance, Ruff | cloud и builder |
| Container Tools | docker compose |
| SQLite Viewer (`qwtel.sqlite-viewer`) | Открывать `.qqstory` и сохранения |
| Radon IDE (по желанию) | Симулятор в VS Code. Для коммерческого проекта платный ($19–29 в месяц), есть пробный период |

## Распространение сборок

- Свой iPhone, бесплатный Apple ID: установка через Xcode, подпись живёт 7 дней.
- TestFlight и раздача партнёру: Apple Developer Program, $99 в год.
- Android: APK устанавливается напрямую.
