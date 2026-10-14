# The Brass Vault

Демо-история: стимпанк-ограбление в портовом городе Vellmoor (1887). Команда Mags Holloway должна в ночь бала основателей вынести Clockheart из хранилища-оррери банка Ashcombe и до рассвета добраться до мачты с дирижаблем.

| Файл | Что | В git |
|---|---|---|
| `lore/` | Исходный текст: мир, 11 локаций, 9 персонажей с персонами | да |
| `graph.json` | Результат `builder extract` (Haiku 5.5, sync) | да |
| `curation.toml` | Ручные правки графа — **привязаны к этому `graph.json`** | да |
| `meta.toml` | `story_id`, название и тексты мира — переживают пересборку (ADR 24) | да |
| `graph.curated.json` | Промежуточный файл после `curate` | нет (производный) |
| `brass-vault.qqstory` | Готовая история: 74 сущности, 127 связей (после нормализации), 148 эмбеддингов (1536) | да |

## Сборка (из `builder/`)

```bash
uv run builder extract ../stories/brass-vault/lore -o ../stories/brass-vault/graph.json   # ~1 мин, ~$0.03
uv run builder curate ../stories/brass-vault/graph.json ../stories/brass-vault/curation.toml \
    -o ../stories/brass-vault/graph.curated.json
uv run builder export ../stories/brass-vault/graph.curated.json -o ../stories/brass-vault/brass-vault.qqstory \
    --start "The Rusty Gear Tavern" --meta ../stories/brass-vault/meta.toml
uv run builder meta ../stories/brass-vault/brass-vault.qqstory --meta ../stories/brass-vault/meta.toml --set "world_constraints=..."   # ручная правка
uv run builder validate ../stories/brass-vault/brass-vault.qqstory
```

- **Повторное извлечение** называет сущности чуть иначе — `curate` перечислит все правила, которые больше не совпадают; правки надо обновить под новый `graph.json`.
- **Тексты мира** живут в `meta.toml` и переживают пересборку: `world_summary` и `style` — черновик LLM без правок, `world_constraints` — переписан по лору (игрок — приезжий специалист, адаптация вместо отказа).

## Что проверено вручную

- От The Rusty Gear Tavern достижимы все места. В Fenn's Clockworks и Founder's Clocktower builder добавил проходы из улицы и площади (ADR 31); здание банка — только группа комнат со своими проходами.
- Персоны NPC полные: Horace знает положения планет для замка-оррери, Silas — комбинацию сейфа (1823), Tobias — три слабости стража, Pip — пневмопочту и путь через канализацию, Bertha — тайник со вторым ключом.
- Инструменты Fenn'а у него в руках (покупка, не подбор); перфокарта в сейфе; 16 предметов помечены `fixed` (люк, сейф, замок, дверь хранилища…).
