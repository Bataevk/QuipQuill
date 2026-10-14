"""stories/<id>/meta.toml: the story's identity and authored texts that survive rebuilds (ADR 24).

`builder export` takes `story_id`, `title`, `language` and the authored meta from here instead of
generating a new id and empty texts; `builder meta` writes drafts and edits back here.
"""

import tomllib
import uuid
from pathlib import Path

from quipquill_builder.storyfile import AUTHORED_META

IDENTITY_KEYS = ("story_id", "title", "language")
KEYS = IDENTITY_KEYS + AUTHORED_META


class StoryMetaError(ValueError):
    pass


def load(path: Path) -> dict[str, str]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise StoryMetaError(f"{path}: {error}") from error
    unknown = sorted(set(data) - set(KEYS))
    if unknown:
        allowed = ", ".join(KEYS)
        raise StoryMetaError(f"{path}: unknown keys {', '.join(unknown)} (allowed: {allowed})")
    not_text = sorted(key for key, value in data.items() if not isinstance(value, str))
    if not_text:
        raise StoryMetaError(f"{path}: values must be strings: {', '.join(not_text)}")
    if not data.get("story_id", "").strip():
        raise StoryMetaError(f"{path}: story_id is required")
    return data


def save(path: Path, values: dict[str, str]) -> None:
    """Write the known keys in a fixed order; long texts as multi-line strings."""
    lines = [
        "# Story identity and authored texts (ADR 24). `builder export --meta` reads this file;",
        "# `builder meta --meta` writes drafts and edits here. Keep story_id stable: saves use it.",
        "",
    ]
    for key in KEYS:
        if key in values:
            lines.append(f"{key} = {_toml_string(values[key])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def new(title: str, language: str) -> dict[str, str]:
    return {"story_id": str(uuid.uuid4()), "title": title, "language": language}


def from_story_meta(meta: dict[str, str]) -> dict[str, str]:
    """The subset of a .qqstory meta table that meta.toml keeps."""
    return {key: meta[key] for key in KEYS if meta.get(key, "").strip()}


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    if "\n" not in value:
        return f'"{escaped}"'
    # Multi-line basic string; the newline right after the opening quotes is trimmed by TOML.
    return f'"""\n{escaped}"""'
