"""Reading and writing .qqstory files (SQLite, schema in schema/story.sql)."""

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from quipquill_builder.story import StoryData

SCHEMA_VERSION = "1"

REQUIRED_META = (
    "schema_version",
    "story_id",
    "title",
    "language",
    "start_location",
    "world_summary",
    "world_constraints",
    "style",
    "embed_model",
    "embed_dim",
    "built_at",
    "builder_version",
)
# Keys written by `builder meta` (LLM draft + manual edits) rather than by `export`.
AUTHORED_META = ("world_summary", "world_constraints", "style")
# Optional authored keys (meta.toml or `builder meta --set`): luck, 0..1, how lucky the world is
# for the player (ADR 47), missing means a fair world, 0.5; tagline, one line under the title on
# the cover; heroes, the story's ready-made player characters as a JSON list (ADR 62; in meta.toml
# they are [[heroes]] tables).
OPTIONAL_META = ("luck", "tagline", "heroes")
# A hero's sheet (ADR 62): `key` is stable across rebuilds (the app keeps `<story_id>:<key>`).
HERO_FIELDS = ("key", "name", "sex", "age", "race", "blurb", "description")


def schema_path() -> Path:
    """schema/story.sql at the repository root; override with STORY_SCHEMA_PATH."""
    override = os.environ.get("STORY_SCHEMA_PATH")
    if override:
        return Path(override)
    # builder/src/quipquill_builder/storyfile.py → repository root is four levels up.
    return Path(__file__).resolve().parents[3] / "schema" / "story.sql"


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create(path: Path, story: StoryData, meta: dict[str, str]) -> None:
    """Create a new story file with entities, relations and meta (no embeddings yet)."""
    if path.exists():
        path.unlink()
    with connect(path) as conn:
        conn.executescript(schema_path().read_text(encoding="utf-8"))
        conn.executemany(
            "INSERT INTO entities (id, type, subtype, name, description, props)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (e.id, e.type, e.subtype, e.name, e.description, json.dumps(e.props))
                for e in story.entities
            ],
        )
        conn.executemany(
            "INSERT INTO relations (src, dst, type, description) VALUES (?, ?, ?, ?)",
            [(r.src, r.dst, r.type, r.description) for r in story.relations],
        )
        set_meta(conn, meta)


def set_meta(conn: sqlite3.Connection, values: dict[str, str]) -> None:
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        list(values.items()),
    )


def read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM meta").fetchall())
