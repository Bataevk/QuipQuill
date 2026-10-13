from dataclasses import asdict, dataclass, field
from typing import Any

GRAPH_FORMAT_VERSION = 1


@dataclass
class Entity:
    name: str
    type: str = "OTHER"
    description: str = ""
    # Free-form flags set during curation, e.g. {"fixed": true} for things that cannot be taken.
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    source: str
    target: str
    type: str = "OTHER"
    description: str = ""


@dataclass
class Graph:
    """Extracted story graph. Entities are referenced by their canonical `name`."""

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": GRAPH_FORMAT_VERSION,
            "meta": self.meta,
            "entities": [asdict(e) for e in self.entities],
            "relationships": [asdict(r) for r in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Graph":
        return cls(
            entities=[Entity(**e) for e in data.get("entities", [])],
            relationships=[Relationship(**r) for r in data.get("relationships", [])],
            meta=dict(data.get("meta", {})),
        )
