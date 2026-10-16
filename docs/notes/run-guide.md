# Как запустить прототип и поиграть (macOS: iOS-симулятор и Android-эмулятор)

Пошагово, от выключенного компьютера до хода в игре. Нужны **два окна терминала**: в одном работает облако, в другом — Metro (сборщик приложения). Оба окна держать открытыми, пока играете.

Состояние машины и подробности — [dev-setup.md](dev-setup.md).

## 0. Один раз

| Что | Как проверить |
|---|---|
| Xcode и iOS Simulator | `xcrun simctl list devices` показывает iPhone |
| Node, uv, CocoaPods, Docker Desktop | см. [dev-setup.md](dev-setup.md#состояние-машины-разработчика-2026-10-15) |
| Зависимости приложения | `cd apps/mobile && npm install` |
| Зависимости облака | `cd cloud && uv sync` |
| `cloud/.env` | скопирован из `cloud/.env.example`; заданы `OPENROUTER_API_KEY`, `JWT_SECRET` (≥ 32 символов) и `INVITE_CODES` — **JSON-список**, например `INVITE_CODES=["my-invite"]` |

## 1. Облако (терминал 1, из корня репозитория)

```bash
open -a Docker
```

Подождать, пока Docker Desktop запустится (иконка кита в меню перестанет «дышать»), затем:

```bash
make infra-up
```

```bash
make cloud-migrate
```

```bash
make cloud-dev
```

- `make infra-up` — Postgres и RabbitMQ; ждёт, пока оба станут healthy.
- `make cloud-migrate` — нужно в первый раз и после новых миграций; повторный запуск безвреден.
- `make cloud-dev` — API на `http://127.0.0.1:8000`. **Запускать после `infra-up`**: если RabbitMQ ещё не поднят, API стартует с сообщением `RabbitMQ unavailable … WebSocket chat is disabled`, и игра будет висеть — тогда остановить (`Ctrl+C`) и запустить снова.

Проверка (в любом третьем окне): все три должны ответить `"status":"ok"`.

```bash
curl -s localhost:8000/health/db; echo; curl -s localhost:8000/health/mq; echo; curl -s localhost:8000/health/llm; echo
```

## 2. Приложение (терминал 2, из `apps/mobile`)

**Первый раз** (и после добавления пакетов с нативным кодом) — собрать dev-клиент. Это 5–10 минут; команда сама откроет симулятор, установит приложение и запустит Metro:

```bash
LANG=en_US.UTF-8 npx expo run:ios
```

**Дальше** пересобирать не нужно — достаточно Metro:

```bash
npx expo start
```

и открыть приложение:

- в окне Metro нажать `i` — откроется симулятор и в нём QuipQuill;
- или вручную: `open -a Simulator`, на экране симулятора нажать иконку **QuipQuill**, в лаунчере dev-клиента выбрать сервер `http://localhost:8081`.

Полезное в окне Metro: `r` — перезагрузить приложение (если правки кода не подхватились), `j` — отладчик. Ошибки JS видны прямо в этом окне.

## 2а. Android-эмулятор (вместо iOS или вместе с ним)

**Один раз:**

- Android Studio установлен; эмулятор есть (`Medium_Phone_API_37.0`; свой — Android Studio → Device Manager).
- JDK **17**: `brew install openjdk@17`. JDK 25 из Android Studio не подходит — сборка нативного кода падает на `WARNING: A restricted method in java.lang.System has been called`.
- Переменные окружения — добавить в `~/.zshrc` и открыть новый терминал:

  ```bash
  export ANDROID_HOME="$HOME/Library/Android/sdk"
  export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
  export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$JAVA_HOME/bin:$PATH"
  ```

**Запустить эмулятор** (или кнопкой ▶ в Device Manager Android Studio):

```bash
emulator -avd Medium_Phone_API_37.0
```

**Первая сборка** dev-клиента (из `apps/mobile`, эмулятор уже запущен) — первый раз 10–20 минут, дальше ~1,5 минуты:

```bash
npx expo run:android
```

Если первая сборка упала на `Downloading … gradle-….zip failed: timeout` — сервер Gradle отвечает медленно; просто повторить команду.

**Дальше** — как с iOS: `npx expo start`, в окне Metro нажать `a`. Можно держать открытыми оба эмулятора с одним Metro.

**Облако** приложение на эмуляторе находит само: `localhost` компьютера для эмулятора — это `10.0.2.2`, адрес подставляется автоматически (`src/platform/services.ts`).

**Импорт истории на Android** — через обычный выбор файла: положить файл в «Загрузки» эмулятора (из корня репозитория)

```bash
adb push stories/brass-vault/brass-vault.qqstory /sdcard/Download/
```

и в приложении **Stories** → **Import story** → `brass-vault.qqstory`.

Особенности эмулятора: при первом вводе текста клавиатура Gboard может показать обучающий экран «Try out your stylus» — нажать **Cancel**. Кнопка «назад» на главном экране закрывает приложение — открыть его снова иконкой.

## 2б. Против тестового сервера (без локального облака)

Шаг 1 не нужен — облако уже работает на VPS ([deploy](deploy.md)). Metro запускать так:

```bash
EXPO_PUBLIC_CLOUD_URL=https://146-103-120-112.sslip.io npx expo start --clear
```

`--clear` нужен при смене адреса: переменная вшивается в бандл. Инвайт-код сервера — см. [deploy](deploy.md#сервер); аккаунты на сервере и локально разные.

## 3. В приложении

1. **Вход.** Email, пароль и инвайт-код из `INVITE_CODES` → **Create account**. В следующие разы — только email и пароль → **Sign in**. Email с доменами `.test`, `.example` облако отклоняет — возьмите любой обычный, письма не отправляются.
2. **Импорт истории.** На Android — см. раздел 2а. На iOS **Stories** → **Import story** открывает системный выбор файла, но в «Файлы» симулятора `.qqstory` просто так не положить. Проще одной командой из корня репозитория (приложение должно быть открыто):

   ```bash
   DATA=$(xcrun simctl get_app_container booted app.quipquill.mobile data); cp stories/brass-vault/brass-vault.qqstory "$DATA/tmp/"; xcrun simctl openurl booted "quipquill://stories?importPath=$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "$DATA/tmp/brass-vault.qqstory")"
   ```

   На экране появится «Imported “The Brass Vault” (sqlite-vec)».
3. **Персонаж.** **Characters** → **New character**: имя, пол, возраст, раса, описание. Можно нарочно сделать того, кого в мире нет (эльф, маг) — посмотреть адаптацию.
4. **Новая игра.** **New game** → выбрать историю и персонажа → **Start**.
   - Сначала адаптация: объяснение и карточки — адаптированный персонаж и до двух альтернатив. Выбрать карточку или написать пожелание внизу (всего до трёх предложений, третье — только выбор).
   - После выбора мастер открывает сцену. Дальше пишите, что делаете: «I ask Mags about the vault», «I take the lantern», «I go up to Lamplight Row».
   - Реплики NPC идут отдельными пузырями с именем. Ход с реакцией NPC занимает 15–20 секунд.
5. **Главный экран.** Партия в списке **Continue**: нажать — продолжить с того же места; **Start over** — начать заново (всё сыгранное теряется); **Delete** — удалить.

## 4. Что важно знать

- **Дневной лимит.** У всех аккаунтов тариф free: $0.05 в UTC-сутки (`cloud/src/quipquill_cloud/llm/policy.py`). Средний ход ≈ $0.0023, то есть ~20 ходов в день. Дальше ходы отклоняются до полуночи по UTC. Для своих прогонов и показа лимит можно поднять для всех, не трогая код: добавить в `cloud/.env` строку `DAILY_COST_LIMIT_USD=1` и перезапустить `make cloud-dev`.
- **Стоимость хода видна прямо в игре** (только dev-сборка): мелкая строка под каждым ходом — цена, число вызовов по каналам (`guard`, `gm`, `npc`), токены и сколько из кэша; внизу ленты — итог за сессию.
- **Автоматический прогон демо-сценария** без телефона (ядро + облако + настоящий Haiku, ~$0.02 за прогон, 3 прогона ≈ 6 минут): `cd apps/mobile && npm run test:live` (нужен `make infra-up`; своё облако тест поднимает сам). Число прогонов — `QQ_RUNS=1 npm run test:live`.
- Потрачено сегодня:

  ```bash
  docker exec quipquill-postgres-1 psql -U quipquill -d quipquill -c "select purpose, count(*), round(sum(cost_usd)::numeric,4) from usage_events where created_at >= date_trunc('day', now() at time zone 'utc') group by purpose;"
  ```

- **Данные приложения** живут в симуляторе: истории, персонажи, партии — в `$(xcrun simctl get_app_container booted app.quipquill.mobile data)/Documents`. Партия — обычный SQLite-файл `saves/<id>.db`, его можно открыть SQLite Viewer'ом (таблицы `transcript`, `events`, `state_*`).
- **Аккаунты** — в Postgres из Docker; `make infra-down` их не стирает (данные в томах).

## 5. Если что-то не так

| Симптом | Причина и что делать |
|---|---|
| Под последним сообщением кнопка «The story paused — continue» появилась сразу при открытии партии | Прошлый ход не закончился — кнопка доигрывает его (начало партии и вступление после выбора персонажа доигрываются сами) |
| Индикатор крутится, под ним «Connecting to the server…» | Облако недоступно: не запущены Docker, `make infra-up` или `make cloud-dev`; либо `make cloud-dev` стартовал раньше RabbitMQ (в его логе `WebSocket chat is disabled`) — тогда перезапустить `make cloud-dev`. Ход не потерян — уйдёт сам, когда облако поднимется |
| При входе «HTTP 500» | Нет Postgres: `make infra-up` |
| «Check the email address…» при регистрации | Домен email зарезервирован (`.test`, `.example`) или пароль слишком короткий |
| «This invite code is not valid» | Код не из `INVITE_CODES` в `cloud/.env` (после правки `.env` перезапустить `make cloud-dev`) |
| Ошибка `limit_exceeded` | Дневной лимит исчерпан — `DAILY_COST_LIMIT_USD` в `cloud/.env` (см. раздел 4) |
| Android: «QuipQuill isn't responding» | Эмулятору не хватает памяти (часто после долгого простоя) — перезапустить эмулятор с холодного старта (Device Manager → ⋮ → Cold Boot Now) |
| Под последним сообщением кнопка «The story paused — continue» | Прошлый ход оборвался (закрыли приложение, сеть) — кнопка доигрывает его |
| Красный экран `Cannot find native module …` | Dev-клиент собран без нового нативного модуля: `LANG=en_US.UTF-8 npx expo run:ios` |
| Android: `A restricted method in java.lang.System has been called` при сборке | Сборка идёт на JDK 25 — проверить `JAVA_HOME` (нужен JDK 17, раздел 2а) и остановить старый демон: `cd android && ./gradlew --stop` |
| Правки кода не видны | В окне Metro `r`; если не помогло — закрыть приложение в симуляторе и открыть снова |
| Хочется начать с чистого листа | Удалить приложение: `xcrun simctl uninstall booted app.quipquill.mobile` (стирает истории, персонажей и партии; аккаунт в облаке остаётся), затем `npx expo run:ios` |

## 6. Выключить

`Ctrl+C` в окнах Metro и облака, затем из корня:

```bash
make infra-down
```

Docker Desktop можно закрыть из меню кита.
