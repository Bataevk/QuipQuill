import sqlite3
from pathlib import Path

import pytest

from quipquill_builder import storyfile
from quipquill_builder.story import StoryData, StoryEntity, StoryRelation
from quipquill_builder.validate import validate

META = {
    "schema_version": "1",
    "story_id": "x",
    "title": "T",
    "language": "en",
    "start_location": "loc:a",
    "world_summary": "w",
    "world_constraints": "c",
    "style": "s",
    "embed_model": "m",
    "embed_dim": "2",
    "built_at": "now",
    "builder_version": "0",
}


@pytest.fixture
def make(tmp_path: Path):
    """Factory for story files built from rows; connections are closed after the test."""
    connections: list[sqlite3.Connection] = []

    def build(entities, relations, **meta) -> sqlite3.Connection:
        path = tmp_path / f"s{len(connections)}.qqstory"
        storyfile.create(path, StoryData(entities, relations), {**META, **meta})
        connections.append(sqlite3.connect(path))
        return connections[-1]

    yield build
    for conn in connections:
        conn.close()


def loc(eid: str) -> StoryEntity:
    return StoryEntity(eid, "LOCATION", "Spot", eid, "a place")


def codes(conn: sqlite3.Connection, level: str) -> set[str]:
    return {i.code for i in validate(conn, check_embeddings=False) if i.level == level}


def test_valid_minimal_story(make) -> None:
    conn = make([loc("loc:a")], [])
    assert codes(conn, "error") == set()


def test_start_location_errors(make) -> None:
    item = StoryEntity("item:x", "ITEM", "Tool", "x", "a thing")
    assert "start_location" in codes(
        make([loc("loc:a"), item], [], start_location="item:x"), "error"
    )
    assert "start_location" in codes(make([loc("loc:a")], [], start_location="loc:zzz"), "error")


def test_one_way_dead_end_is_reachable_but_return_is_not_required(make) -> None:
    conn = make(
        [loc("loc:a"), loc("loc:b")],
        [StoryRelation("loc:a", "loc:b", "LEADS_TO", "one-way drop")],
    )
    assert "unreachable" not in codes(conn, "error")


def test_warnings(make) -> None:
    lost = StoryEntity("item:lost", "ITEM", "Tool", "lost", "")
    coin = StoryEntity("item:coin", "ITEM", "Tool", "coin", "shiny")
    conn = make(
        [loc("loc:a"), loc("loc:b"), lost, coin],
        [
            StoryRelation("loc:a", "loc:b", "LEADS_TO"),
            StoryRelation("item:coin", "loc:a", "LOCATED_IN"),
            StoryRelation("item:coin", "loc:b", "LOCATED_IN"),
            StoryRelation("item:coin", "item:lost", "GLOWS_NEAR"),
            StoryRelation("item:coin", "loc:a", "LEADS_TO"),
        ],
    )
    assert codes(conn, "warning") >= {
        "no_description",
        "unplaced_item",
        "many_places",
        "unknown_relation",
        "path_not_location",
    }


def test_embedding_checks(make) -> None:
    conn = make([loc("loc:a")], [])
    errors = {i.code for i in validate(conn) if i.level == "error"}
    assert "embeddings_missing" in errors

    conn.execute("INSERT INTO embeddings VALUES ('loc:a', 'name', ?)", (b"\x00" * 12,))
    errors = {i.code for i in validate(conn) if i.level == "error"}
    assert errors == {"embeddings_dim"}


@pytest.mark.parametrize("key", ["title", "embed_model", "start_location"])
def test_missing_required_meta(make, key: str) -> None:
    conn = make([loc("loc:a")], [], **{key: ""})
    issues = validate(conn)
    assert any(i.code == "meta_missing" and f"'{key}'" in i.message for i in issues)


def test_draft_mode_downgrades_authored_meta(make) -> None:
    conn = make([loc("loc:a")], [], style="")
    issues = validate(conn, check_embeddings=False, require_authored_meta=False)
    assert [(i.level, i.code) for i in issues] == [("warning", "meta_missing")]
