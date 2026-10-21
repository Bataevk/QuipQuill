"""The story catalogue (ADR 61): every `.qqstory` under STORIES_DIR, described from its `meta`.

The cloud only hands the files out; the world lives on the device, which imports a downloaded
file as before (ADR 1). The folder is scanned again when its files change (a deploy copies new
ones in), like the prompts (llm/prompts.py), so no restart is needed.
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# The .qqstory format this cloud knows (schema/story.sql, `meta.schema_version`).
SCHEMA_VERSION = "1"
REQUIRED = ("schema_version", "story_id", "title")
CHUNK = 1 << 20


@dataclass(frozen=True)
class StoryEntry:
    """What the app shows before downloading; `version` changes with every new build of the file."""

    id: str
    version: str  # sha256 of the file
    title: str
    tagline: str
    summary: str
    heroes: list[dict[str, Any]]
    built_at: str
    size: int
    path: Path = field(repr=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _heroes(raw: str | None) -> list[dict[str, Any]]:
    """`meta.heroes` (ADR 62): a JSON list of sheets; anything else is no heroes, not an error."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except ValueError:
        return []
    return [hero for hero in value if isinstance(hero, dict)] if isinstance(value, list) else []


def read_entry(path: Path) -> StoryEntry:
    """Raises ValueError (or sqlite3.Error) for a file that is not a story this cloud can offer."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    try:
        meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
    finally:
        connection.close()
    missing = [key for key in REQUIRED if not str(meta.get(key) or "").strip()]
    if missing:
        raise ValueError(f"missing meta: {', '.join(missing)}")
    if str(meta["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"schema_version {meta['schema_version']}")
    return StoryEntry(
        id=str(meta["story_id"]),
        version=_sha256(path),
        title=str(meta["title"]),
        tagline=str(meta.get("tagline") or ""),
        summary=str(meta.get("world_summary") or ""),
        heroes=_heroes(meta.get("heroes")),
        built_at=str(meta.get("built_at") or ""),
        size=path.stat().st_size,
        path=path,
    )


class StoryCatalog:
    """The stories under `root` (searched recursively), rescanned when the files change."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = Lock()
        self._signature: tuple[tuple[str, int, int], ...] | None = None
        self._entries: dict[str, StoryEntry] = {}

    def _files(self) -> list[Path]:
        if not self._root.is_dir():
            return []
        return sorted(p for p in self._root.rglob("*.qqstory") if p.is_file())

    def entries(self) -> list[StoryEntry]:
        files = self._files()
        signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        with self._lock:
            if signature != self._signature:
                self._entries = self._scan(files)
                self._signature = signature
            return sorted(self._entries.values(), key=lambda entry: entry.title.lower())

    def get(self, story_id: str) -> StoryEntry | None:
        return next((entry for entry in self.entries() if entry.id == story_id), None)

    @staticmethod
    def _scan(files: list[Path]) -> dict[str, StoryEntry]:
        found: dict[str, StoryEntry] = {}
        for path in files:
            try:
                entry = read_entry(path)
            except (ValueError, sqlite3.Error, OSError) as error:
                logger.warning("Skipping %s: not a playable story (%s)", path.name, error)
                continue
            current = found.get(entry.id)
            if current is not None:
                # Two files of one story: offer the newer build.
                logger.warning("Story %s is in %s and %s", entry.id, current.path.name, path.name)
                if current.built_at >= entry.built_at:
                    continue
            found[entry.id] = entry
        return found
