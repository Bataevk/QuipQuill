import asyncio
from pathlib import Path

import pytest
from conftest import FakeChatClient

from quipquill_builder.extractor import ExtractionError, ExtractionParams, GraphExtractor
from quipquill_builder.models import Graph
from quipquill_builder.text import Document, load_documents

# Small chunks so that each fixture section becomes its own chunk.
PARAMS = ExtractionParams(chunk_size=600, chunk_overlap=0, merge_threshold=0.85)


def extract(client: FakeChatClient, documents: list[Document]) -> Graph:
    return asyncio.run(GraphExtractor(client, params=PARAMS).extract(documents))


def test_extracts_merged_graph(fake_client: FakeChatClient, lore_dir: Path) -> None:
    graph = extract(fake_client, load_documents(lore_dir))

    entities = {e.name: e for e in graph.entities}
    assert set(entities) == {
        "The Entrance Rune Hall",
        "copper torch",
        "Torchlight Corridor",
        "Skeleton Guardian",
        "Whispering Woods",
    }
    # "Torchlight Corridor." from the second chunk merged into the first spelling and the
    # two descriptions were summarized into one.
    assert entities["Torchlight Corridor"].description.startswith("Summary:")
    assert "soot-blackened" in entities["Torchlight Corridor"].description
    # Referenced only by a relationship → added with an empty description.
    assert entities["Whispering Woods"].type == "OTHER"
    assert entities["Whispering Woods"].description == ""

    edges = {(r.source, r.type, r.target) for r in graph.relationships}
    assert edges == {
        ("copper torch", "LOCATED_IN", "The Entrance Rune Hall"),
        ("The Entrance Rune Hall", "LEADS_TO", "Torchlight Corridor"),
        ("Skeleton Guardian", "LOCATED_IN", "Torchlight Corridor"),
        ("Torchlight Corridor", "LEADS_TO", "The Entrance Rune Hall"),
        ("Torchlight Corridor", "LEADS_TO", "Whispering Woods"),
    }
    assert graph.meta["chunks"] == 2
    assert graph.meta["failed_chunks"] == 0


def test_bad_chunk_is_skipped_and_counted(lore_dir: Path) -> None:
    client = FakeChatClient(
        {
            "The Entrance Rune Hall**": '{"entities": [{"name": "copper torch"}]}',
            "Torchlight Corridor**": "Sorry, I cannot help with that.",
        }
    )
    graph = extract(client, load_documents(lore_dir))

    assert [e.name for e in graph.entities] == ["copper torch"]
    assert graph.meta["failed_chunks"] == 1


def test_all_chunks_failing_raises(lore_dir: Path) -> None:
    client = FakeChatClient({"**": "not json"})
    with pytest.raises(ExtractionError):
        extract(client, load_documents(lore_dir))


def test_empty_input_raises() -> None:
    with pytest.raises(ExtractionError):
        extract(FakeChatClient({}), [])


def test_graph_round_trips_through_dict(fake_client: FakeChatClient, lore_dir: Path) -> None:
    graph = extract(fake_client, load_documents(lore_dir))
    data = graph.to_dict()

    assert data["version"] == 1
    assert Graph.from_dict(data) == graph
