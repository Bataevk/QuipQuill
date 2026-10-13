import json
from pathlib import Path

import pytest

from quipquill_builder import prompts
from quipquill_builder.models import Entity, Graph, Relationship

FIXTURES = Path(__file__).parent / "fixtures"

# Canned extraction replies, keyed by a phrase that identifies the chunk.
HALL_REPLY = {
    "entities": [
        {
            "name": "The Entrance Rune Hall",
            "type": "Building",
            "description": "A vaulted hall lit by faintly glowing runes.",
        },
        {"name": "copper torch", "type": "Tool", "description": "Rests on a low pedestal."},
        {"name": "Torchlight Corridor", "type": "Building", "description": "Lies to the north."},
    ],
    "relationships": [
        {
            "source_entity_name": "copper torch",
            "relationship_type": "LOCATED_IN",
            "context_phrase": "rests on a low pedestal",
            "target_entity_name": "The Entrance Rune Hall",
        },
        {
            "source_entity_name": "The Entrance Rune Hall",
            "relationship_type": "LEADS_TO",
            "context_phrase": "North → Torchlight Corridor",
            "target_entity_name": "Torchlight Corridor",
        },
    ],
}

CORRIDOR_REPLY = {
    "entities": [
        {
            "name": "Torchlight Corridor.",
            "type": "Building",
            "description": "A long corridor lined with soot-blackened sconces.",
        },
        {
            "name": "Skeleton Guardian",
            "type": "Creature",
            "description": "Slumbers in the semi-darkness. HP 12, atk 3, def 2.",
        },
    ],
    "relationships": [
        {
            "source_entity_name": "Skeleton Guardian",
            "relationship_type": "located_in",
            "context_phrase": "slumbers in the semi-darkness",
            "target_entity_name": "torchlight corridor",
        },
        {
            "source_entity_name": "Torchlight Corridor",
            "relationship_type": "LEADS_TO",
            "context_phrase": "South → The Entrance Rune Hall",
            "target_entity_name": "Entrance Rune Hall",
        },
        {
            "source_entity_name": "Torchlight Corridor",
            "relationship_type": "LEADS_TO",
            "context_phrase": "East → Whispering Woods",
            "target_entity_name": "Whispering Woods",
        },
    ],
}


class FakeChatClient:
    """Deterministic stand-in for OpenRouter: no network in tests."""

    def __init__(self, replies: dict[str, str], model: str = "fake/model") -> None:
        self.model = model
        self.replies = replies
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if system in (prompts.SUMMARIZATION_SYSTEM, prompts.PERSONA_MERGE_SYSTEM):
            return "Summary: " + " ".join(user.splitlines()[1:])
        for marker, reply in self.replies.items():
            if marker in user:
                return reply
        raise AssertionError(f"Unexpected chunk: {user[:80]!r}")


@pytest.fixture(autouse=True)
def _isolate_from_local_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tests never read builder/.env (real key, local switches) and never reach the network."""
    monkeypatch.chdir(tmp_path)
    for name in ("OPENROUTER_API_KEY", "USE_BATCH", "PROVIDER_ONLY", "EXTRACT_MODEL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def lore_dir() -> Path:
    return FIXTURES / "lore"


@pytest.fixture
def fake_client() -> FakeChatClient:
    return FakeChatClient(
        {
            "The Entrance Rune Hall**": "```json\n" + json.dumps(HALL_REPLY) + "\n```",
            "Torchlight Corridor**": json.dumps(CORRIDOR_REPLY),
        }
    )


class FakeEmbedder:
    """Deterministic 8-dimensional vectors derived from the text hash."""

    def __init__(self, model: str = "fake/embed", dim: int = 8) -> None:
        self.model = model
        self.dim = dim
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [
            [((hash(text) >> (i * 4)) % 1000) / 1000 for i in range(self.dim)] for text in texts
        ]


def story_graph() -> Graph:
    """A small extracted graph that exercises every normalization rule."""
    E, R = Entity, Relationship
    return Graph(
        entities=[
            E("The Entrance Rune Hall", "Building", "A vaulted hall lit by glowing runes."),
            E("copper torch", "Tool", "Shines 5m."),
            E("Torchlight Corridor", "Building", "A long soot-blackened corridor."),
            E("Skeleton Guardian", "Creature", "HP 12, atk 3, def 2."),
            E("rusty key", "Tool", "Opens simple locks."),
            E("Whispering Woods", "OTHER", ""),
            E("Collapsed Shaft", "Spot", "A pit with a rope ladder."),
        ],
        relationships=[
            R("The Entrance Rune Hall", "copper torch", "HAS_ITEM", "What to take"),
            R("The Entrance Rune Hall", "Torchlight Corridor", "LEADS_TO", "North"),
            R("Torchlight Corridor", "Skeleton Guardian", "HAS_AGENT", "slumbers here"),
            R("Skeleton Guardian", "rusty key", "HAS_ITEM", "on its belt"),
            R("Torchlight Corridor", "Whispering Woods", "LEADS_TO", "East"),
            R("Torchlight Corridor", "Collapsed Shaft", "LEADS_TO", "a one-way drop"),
            R("Collapsed Shaft", "The Entrance Rune Hall", "LEADS_TO", "rope ladder up"),
        ],
        meta={"extract_model": "fake/model"},
    )


@pytest.fixture
def graph() -> Graph:
    return story_graph()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
