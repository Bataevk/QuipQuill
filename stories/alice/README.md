# Alice's Adventures in Wonderland

Lewis Carroll, 1865 — public domain. Lore: the Project Gutenberg text (eBook #11) without the Gutenberg header, chapters I–XII (`lore/alice.txt`). The built-in story of the app since ADR 80.

Playable map (16 places): Riverbank → The Rabbit-Hole → Hall of Doors → Pool of Tears → Shore of the Pool → The Wood → White Rabbit's House (Tidy Little Room), Caterpillar's Mushroom, Duchess's House (Duchess's Kitchen), March Hare's House, Tree with a Door → Hall of Doors → The Queen's Garden → Sea-Shore, Court of Justice. Hero: Alice (`meta.toml`).

## Rebuild (from `builder/`)

```bash
uv run builder extract ../stories/alice/lore -o ../stories/alice/graph.json --sync   # ~$0.5, only if the lore changes
# curation.toml is generated: merges, renames, map, placements and descriptions live in gen_curation.py
python3 ../stories/alice/gen_curation.py ../stories/alice                       # pass 1 (no placement fixes)
uv run builder curate ../stories/alice/graph.json ../stories/alice/curation.toml -o /tmp/alice-pass1.json
python3 ../stories/alice/gen_curation.py ../stories/alice /tmp/alice-pass1.json  # pass 2: drops the extractor's placements
uv run builder curate ../stories/alice/graph.json ../stories/alice/curation.toml -o ../stories/alice/graph.curated.json
uv run builder export ../stories/alice/graph.curated.json -o ../stories/alice/alice.qqstory --start "Riverbank" --meta ../stories/alice/meta.toml --sync
uv run builder validate ../stories/alice/alice.qqstory
```
