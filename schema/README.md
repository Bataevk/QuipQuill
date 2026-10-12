# schema

[`story.sql`](story.sql) — единственный источник схемы файла `.qqstory` (версия 1), общий для builder (Python) и мобильного ядра (TypeScript). Описание и соглашения — [data-model](../docs/architecture/data-model.md).

Изменение схемы = новое значение `meta.schema_version`, обновление валидатора в builder и чтения в ядре.
