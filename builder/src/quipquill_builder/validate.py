"""Checks that a .qqstory file is playable. Errors block export; warnings are reported."""

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from quipquill_builder.storyfile import (
    AUTHORED_META,
    HERO_FIELDS,
    REQUIRED_META,
    SCHEMA_VERSION,
    read_meta,
)

KNOWN_RELATIONS = {"LOCATED_IN", "HAS_ITEM", "LEADS_TO", "HAS_ATTRIBUTE", "INTERACTS_WITH"}
PORTABLE_TYPES = ("ITEM", "DOCUMENT")


@dataclass(frozen=True)
class Issue:
    level: Literal["error", "warning"]
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7} {self.code}: {self.message}"


def validate(
    conn: sqlite3.Connection,
    *,
    check_embeddings: bool = True,
    require_authored_meta: bool = True,
) -> list[Issue]:
    issues: list[Issue] = []

    def error(code: str, message: str) -> None:
        issues.append(Issue("error", code, message))

    def warning(code: str, message: str) -> None:
        issues.append(Issue("warning", code, message))

    meta = read_meta(conn)
    entities = {
        row[0]: (row[1], row[2], row[3])
        for row in conn.execute("SELECT id, type, name, description FROM entities")
    }
    relations = conn.execute("SELECT src, dst, type, description FROM relations").fetchall()

    # --- meta
    if meta.get("schema_version") != SCHEMA_VERSION:
        error("schema_version", f"expected {SCHEMA_VERSION}, got {meta.get('schema_version')!r}")
    skipped = set() if check_embeddings else {"embed_model", "embed_dim"}
    for key in REQUIRED_META:
        if key in skipped or meta.get(key, "").strip():
            continue
        if key in AUTHORED_META and not require_authored_meta:
            warning("meta_missing", f"'{key}' is empty — fill it with `builder meta`")
        else:
            error("meta_missing", f"required meta key '{key}' is empty")
    luck = meta.get("luck", "").strip()
    if luck and not _is_luck(luck):
        error("meta_luck", f"luck must be a number from 0 to 1, got {luck!r}")
    if "\n" in meta.get("tagline", ""):
        error("meta_tagline", "tagline must be one line")
    for problem in _hero_problems(meta.get("heroes", "")):
        error("meta_heroes", problem)

    # --- entities
    locations = {eid for eid, (etype, _, _) in entities.items() if etype == "LOCATION"}
    if not entities:
        error("no_entities", "the story has no entities")
    elif not locations:
        error("no_locations", "the story has no locations")
    for eid, (_, name, description) in entities.items():
        if not description.strip():
            warning("no_description", f"{eid} ('{name}') has no description")

    start = meta.get("start_location", "")
    if start and start not in entities:
        error("start_location", f"start location {start!r} does not exist")
    elif start and start not in locations:
        error("start_location", f"start location {start!r} is not a LOCATION")

    # --- relations
    exits: dict[str, set[str]] = defaultdict(set)
    placed: dict[str, list[str]] = defaultdict(list)
    held: set[str] = set()
    for src, dst, rel_type, _ in relations:
        if src not in entities or dst not in entities:
            error("dangling_relation", f"{src} -{rel_type}-> {dst} points to a missing entity")
            continue
        if rel_type not in KNOWN_RELATIONS:
            warning("unknown_relation", f"{src} -{rel_type}-> {dst} has an unknown type")
        if rel_type == "LEADS_TO":
            if src in locations and dst in locations:
                exits[src].add(dst)
            else:
                warning("path_not_location", f"{src} -LEADS_TO-> {dst} connects a non-location")
        elif rel_type == "LOCATED_IN":
            placed[src].append(dst)
        elif rel_type == "HAS_ITEM":
            held.add(dst)

    for eid, places in placed.items():
        if len(places) > 1:
            warning("many_places", f"{eid} is LOCATED_IN several places: {', '.join(places)}")
    for eid, (etype, name, _) in entities.items():
        if etype in PORTABLE_TYPES and eid not in placed and eid not in held:
            warning("unplaced_item", f"{eid} ('{name}') is not located anywhere")

    if start in locations:
        inside = {eid: places for eid, places in placed.items() if eid in locations}
        reachable = _reachable(start, exits, inside)
        unreachable = sorted(locations - reachable)
        if unreachable:
            error(
                "unreachable",
                f"locations not reachable from {start} via LEADS_TO "
                f"(or as part of a reachable place): {', '.join(unreachable)}",
            )

    # --- embeddings
    if check_embeddings:
        issues.extend(_check_embeddings(conn, meta, entities.keys()))
    return issues


def _reachable(start: str, exits: dict[str, set[str]], inside: dict[str, list[str]]) -> set[str]:
    """Places reachable by paths, plus places that are part of a reachable place (ADR 19)."""
    seen, stack = {start}, [start]
    while True:
        while stack:
            for nxt in exits.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        parts = [p for p, parents in inside.items() if p not in seen and seen & set(parents)]
        if not parts:
            return seen
        seen.update(parts)
        stack.extend(parts)


def _check_embeddings(conn: sqlite3.Connection, meta: dict[str, str], ids) -> list[Issue]:
    issues = []
    try:
        dim = int(meta.get("embed_dim", ""))
    except ValueError:
        return [
            Issue("error", "embed_dim", f"embed_dim is not a number: {meta.get('embed_dim')!r}")
        ]
    with_name = {
        row[0] for row in conn.execute("SELECT entity_id FROM embeddings WHERE kind = 'name'")
    }
    missing = sorted(set(ids) - with_name)
    if missing:
        issues.append(
            Issue("error", "embeddings_missing", f"{len(missing)} entities lack a name embedding")
        )
    bad = conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE length(vector) != ?", (dim * 4,)
    ).fetchone()[0]
    if bad:
        issues.append(Issue("error", "embeddings_dim", f"{bad} vectors are not {dim}-dimensional"))
    return issues


def has_errors(issues: list[Issue]) -> bool:
    return any(issue.level == "error" for issue in issues)


HERO_KEY = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _hero_problems(raw: str) -> list[str]:
    """The story's heroes (ADR 62): a JSON list of sheets with a unique key, a name and a text."""
    if not raw.strip():
        return []
    try:
        heroes = json.loads(raw)
    except ValueError as error:
        return [f"heroes is not JSON: {error}"]
    if not isinstance(heroes, list):
        return ["heroes must be a list"]
    problems: list[str] = []
    seen: set[str] = set()
    for i, hero in enumerate(heroes, 1):
        if not isinstance(hero, dict):
            problems.append(f"hero #{i} is not an object")
            continue
        key = hero.get("key")
        label = f"hero #{i} ({key or hero.get('name') or '?'})"
        if not isinstance(key, str) or not HERO_KEY.match(key):
            problems.append(f"{label}: key must be lowercase letters, digits and dashes")
        elif key in seen:
            problems.append(f"{label}: key is used twice")
        else:
            seen.add(key)
        for field in ("name", "description"):
            if not isinstance(hero.get(field), str) or not hero[field].strip():
                problems.append(f"{label}: {field} is required")
        for field in ("sex", "race", "blurb"):
            if field in hero and not isinstance(hero[field], str):
                problems.append(f"{label}: {field} must be text")
        age = hero.get("age")
        if age is not None and (isinstance(age, bool) or not isinstance(age, int) or age <= 0):
            problems.append(f"{label}: age must be a whole number above 0")
        unknown = sorted(set(hero) - set(HERO_FIELDS))
        if unknown:
            problems.append(f"{label}: unknown fields {', '.join(unknown)}")
    return problems


def _is_luck(value: str) -> bool:
    try:
        return 0.0 <= float(value) <= 1.0
    except ValueError:
        return False
