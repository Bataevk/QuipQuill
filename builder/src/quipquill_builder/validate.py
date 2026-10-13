"""Checks that a .qqstory file is playable. Errors block export; warnings are reported."""

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from quipquill_builder.storyfile import AUTHORED_META, REQUIRED_META, SCHEMA_VERSION, read_meta

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
