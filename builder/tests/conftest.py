import json
from pathlib import Path

import pytest

from quipquill_builder import prompts

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
        if system == prompts.SUMMARIZATION_SYSTEM:
            return "Summary: " + " ".join(user.splitlines()[1:])
        for marker, reply in self.replies.items():
            if marker in user:
                return reply
        raise AssertionError(f"Unexpected chunk: {user[:80]!r}")


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
