"""Reproducible manual fixes for an extracted graph, kept in a TOML file next to the story.

Applied in this order (names in later steps refer to entities after the merges):
  [merge]            "Canonical" = ["alias", ...]  — aliases' relations move to the canonical
  [drop]             entities = [...]               — removed together with their relations
  [rename]           "old name" = "New Name"        — the name players and the master see
  [retype]           "name" = "Tool"                — extractor type (prompts.ENTITY_TYPES)
  [describe]         "name" = "new description"
  [props]            "name" = { fixed = true }      — merged into the entity's props
  [[drop_relation]]  source, target, type (optional: all types)
  [[add_relation]]   source, target, type, description

Every name must exist at its step; a typo or a name the extractor no longer produces is an
error, not a silent no-op.
"""

import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from quipquill_builder.models import Graph, Relationship

KNOWN_SECTIONS = {
    "merge",
    "drop",
    "rename",
    "retype",
    "describe",
    "props",
    "drop_relation",
    "add_relation",
}


class CurationError(ValueError):
    pass


def load(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        data = tomllib.load(file)
    unknown = set(data) - KNOWN_SECTIONS
    if unknown:
        raise CurationError(f"Unknown sections in {path.name}: {', '.join(sorted(unknown))}")
    return data


def apply(graph: Graph, rules: dict[str, Any]) -> Graph:
    entities = {e.name: replace(e, props=dict(e.props)) for e in graph.entities}
    relationships = [replace(r) for r in graph.relationships]
    missing: list[str] = []

    def require(name: str, step: str) -> bool:
        if name in entities:
            return True
        missing.append(f"[{step}] {name!r}")
        return False

    # merge
    alias_of: dict[str, str] = {}
    for canonical, aliases in rules.get("merge", {}).items():
        if not require(canonical, "merge"):
            continue
        for alias in aliases:
            if not require(alias, "merge"):
                continue
            merged = entities.pop(alias)
            target = entities[canonical]
            if not target.description and merged.description:
                target.description = merged.description
            target.props = {**merged.props, **target.props}
            alias_of[alias] = canonical
    relationships = [
        replace(r, source=alias_of.get(r.source, r.source), target=alias_of.get(r.target, r.target))
        for r in relationships
    ]

    # drop
    dropped = set()
    for name in rules.get("drop", {}).get("entities", []):
        if require(name, "drop"):
            del entities[name]
            dropped.add(name)
    relationships = [
        r for r in relationships if r.source not in dropped and r.target not in dropped
    ]

    # rename: later steps use the new names
    renamed: dict[str, str] = {}
    for old, new in rules.get("rename", {}).items():
        if not require(old, "rename"):
            continue
        if new in entities:
            missing.append(f"[rename] {old!r} → {new!r}: the new name is taken")
            continue
        entity = entities.pop(old)
        entity.name = new
        entities[new] = entity
        renamed[old] = new
    relationships = [
        replace(r, source=renamed.get(r.source, r.source), target=renamed.get(r.target, r.target))
        for r in relationships
    ]

    # retype / describe / props
    for name, new_type in rules.get("retype", {}).items():
        if require(name, "retype"):
            entities[name].type = new_type
    for name, description in rules.get("describe", {}).items():
        if require(name, "describe"):
            entities[name].description = description.strip()
    for name, props in rules.get("props", {}).items():
        if require(name, "props"):
            entities[name].props.update(props)

    # relations
    for rule in rules.get("drop_relation", []):
        source, target, rel_type = rule["source"], rule["target"], rule.get("type")
        before = len(relationships)
        relationships = [
            r
            for r in relationships
            if not (
                r.source == source
                and r.target == target
                and (rel_type is None or r.type == rel_type)
            )
        ]
        if len(relationships) == before:
            missing.append(f"[drop_relation] {source!r} -{rel_type or '*'}-> {target!r}")
    for rule in rules.get("add_relation", []):
        if require(rule["source"], "add_relation") and require(rule["target"], "add_relation"):
            relationships.append(
                Relationship(
                    source=rule["source"],
                    target=rule["target"],
                    type=rule["type"],
                    description=rule.get("description", ""),
                )
            )

    if missing:
        raise CurationError(
            "Curation refers to things not in the graph:\n  " + "\n  ".join(missing)
        )

    unique: dict[tuple[str, str, str], Relationship] = {}
    for r in relationships:
        if r.source == r.target:
            continue
        existing = unique.setdefault((r.source, r.target, r.type), r)
        if not existing.description and r.description:
            existing.description = r.description
    return Graph(
        entities=list(entities.values()),
        relationships=list(unique.values()),
        meta={**graph.meta, "curated": True},
    )
