"""Graph (extractor output) → story rows ready for the .qqstory file.

Responsibilities: map extractor types to story types, assign stable ids, and normalize
relations to the single form documented in schema/story.sql.
"""

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from quipquill_builder.models import Graph
from quipquill_builder.text import normalize_name

logger = logging.getLogger(__name__)

# Extractor type (case-insensitive) → story entity type.
TYPE_MAP = {
    "spot": "LOCATION",
    "building": "LOCATION",
    "country": "LOCATION",
    "location": "LOCATION",
    "person": "NPC",
    "agent": "NPC",
    "creature": "CREATURE",
    "organization": "ORGANIZATION",
    "tool": "ITEM",
    "vehicle": "ITEM",
    "item": "ITEM",
    "document": "DOCUMENT",
}

ID_PREFIX = {
    "LOCATION": "loc",
    "ITEM": "item",
    "NPC": "npc",
    "CREATURE": "npc",
    "ORGANIZATION": "org",
    "DOCUMENT": "doc",
    "OTHER": "ent",
}

ONE_WAY_RE = re.compile(r"\bone[- ]way\b", re.IGNORECASE)


@dataclass
class StoryEntity:
    id: str
    type: str
    subtype: str | None
    name: str
    description: str
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoryRelation:
    src: str
    dst: str
    type: str
    description: str = ""


@dataclass
class StoryData:
    entities: list[StoryEntity] = field(default_factory=list)
    relations: list[StoryRelation] = field(default_factory=list)

    def entity(self, entity_id: str) -> StoryEntity | None:
        return next((e for e in self.entities if e.id == entity_id), None)

    def find(self, name_or_id: str) -> StoryEntity | None:
        """Look up an entity by id, exact name or normalized name."""
        key = normalize_name(name_or_id)
        for entity in self.entities:
            if entity.id == name_or_id or entity.name == name_or_id:
                return entity
        return next((e for e in self.entities if normalize_name(e.name) == key), None)


def slugify(name: str) -> str:
    # Articles are dropped ("The Hall" → "hall") unless nothing else is left ("A" → "a").
    ascii_name = unicodedata.normalize("NFKD", normalize_name(name) or name.lower())
    ascii_name = ascii_name.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-") or "entity"


def story_type(extractor_type: str) -> str:
    return TYPE_MAP.get(extractor_type.strip().lower(), "OTHER")


def build_story(graph: Graph) -> StoryData:
    by_name = {e.name: e for e in graph.entities}
    types = {e.name: story_type(e.type) for e in graph.entities}

    # Untyped endpoints of LEADS_TO are places (e.g. a destination only named in an exit).
    for rel in graph.relationships:
        if rel.type == "LEADS_TO":
            for name in (rel.source, rel.target):
                if types.get(name) == "OTHER":
                    types[name] = "LOCATION"

    ids: dict[str, str] = {}
    used: set[str] = set()
    story = StoryData()
    for entity in graph.entities:
        entity_type = types[entity.name]
        base = f"{ID_PREFIX[entity_type]}:{slugify(entity.name)}"
        entity_id, n = base, 2
        while entity_id in used:
            entity_id, n = f"{base}-{n}", n + 1
        used.add(entity_id)
        ids[entity.name] = entity_id
        subtype = entity.type if entity.type and entity.type.upper() != "OTHER" else None
        story.entities.append(
            StoryEntity(
                entity_id, entity_type, subtype, entity.name, entity.description, entity.props
            )
        )

    relations: dict[tuple[str, str, str], StoryRelation] = {}

    def add(src: str, dst: str, rel_type: str, description: str) -> None:
        if src == dst:
            return
        key = (src, dst, rel_type)
        existing = relations.get(key)
        if existing is None:
            relations[key] = StoryRelation(src, dst, rel_type, description)
        elif not existing.description and description:
            existing.description = description

    for rel in graph.relationships:
        if rel.source not in by_name or rel.target not in by_name:
            logger.warning("Dropping relation with unknown endpoint: %s", rel)
            continue
        src, dst = ids[rel.source], ids[rel.target]
        # One way to say "X is here": location HAS_ITEM / HAS_AGENT → X LOCATED_IN location.
        # An NPC's HAS_ITEM stays as is: that is inventory.
        if rel.type == "HAS_AGENT" or (rel.type == "HAS_ITEM" and types[rel.source] == "LOCATION"):
            add(dst, src, "LOCATED_IN", rel.description)
        else:
            add(src, dst, rel.type, rel.description)

    # Paths are two-way unless the text says otherwise.
    for (src, dst, rel_type), rel in list(relations.items()):
        if rel_type == "LEADS_TO" and not ONE_WAY_RE.search(rel.description):
            add(dst, src, "LEADS_TO", "")

    _link_enclosed_places(story, relations, add)

    story.relations = list(relations.values())
    return story


def _link_enclosed_places(
    story: StoryData,
    relations: dict[tuple[str, str, str], StoryRelation],
    add: Callable[[str, str, str, str], None],
) -> None:
    """A place inside a place with no passages of its own gets a way in and out (ADR 31).

    The extractor often says "Fenn's Clockworks LOCATED_IN Brassworks Alley" and nothing else:
    without a LEADS_TO the player could never enter it, and the NPCs and items in it would be
    out of reach. A place that holds other places (a building whose rooms have their own
    passages) is only a grouping and is left alone.
    """
    names = {e.id: e.name for e in story.entities}
    places = {e.id for e in story.entities if e.type == "LOCATION"}
    with_passage = {end for (src, dst, t) in relations if t == "LEADS_TO" for end in (src, dst)}
    enclosed = [
        (src, dst)
        for (src, dst, t) in relations
        if t == "LOCATED_IN" and src in places and dst in places
    ]
    groupings = {parent for _, parent in enclosed}
    for child, parent in enclosed:
        if child in with_passage or child in groupings:
            continue
        add(parent, child, "LEADS_TO", f"into {names[child]}")
        add(child, parent, "LEADS_TO", f"back out to {names[parent]}")
        logger.info("Linked enclosed place %s ↔ %s", parent, child)
