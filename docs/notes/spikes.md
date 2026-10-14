# Технические проверки (spikes)

Результаты проверок рисков. Каждая запись: что проверяли, на чём (устройство, версии), результат, решение (со ссылкой на пункт в [decisions.md](../architecture/decisions.md)).

## 3.1a op-sqlite + sqlite-vec на устройстве
_Не проведена._

## 3.1b LangChain JS v1 `createAgent` на Hermes
_Не проведена._

## 2.3 Кэширование промпта и effort через OpenRouter
**Проведена 2026-10-14** живыми запросами к `anthropic/claude-haiku-5.5` через OpenRouter (системный промпт ~2 600 токенов с `cache_control: ephemeral`).

- **Кэш:** на Bedrock второй вызов с тем же префиксом читает кэш (`cached_tokens = 2603`, стоимость $0.00003 вместо $0.00033). На Vertex global каждый вызов снова **пишет** кэш (+25%), чтение нестабильно. Без `cache_control` кэша нет ни у кого.
- **Порядок провайдеров:** без `order` OpenRouter выбирал Google, и кэш читался лишь на 4-м вызове; с `order: [amazon-bedrock, google-vertex/global]` — со 2-го.
- **Effort:** `reasoning.effort` `low` и `medium` принимаются; на простых запросах `reasoning_tokens = 0`.
- **Открытый риск:** в сквозном прогоне через Bedrock ~10 с до первого токена (через Vertex ~1 с) — замерить в 6.1.

**Решение:** Bedrock первым, точки кэширования ставит клиент — [ADR 20](../architecture/decisions.md#20-llm-шлюз-лимиты-effort-провайдеры-с-работающим-кэшем). Подробные факты — [research.md](research.md#openrouter-для-игровых-вызовов-2026-10-14).
