import asyncio
import sqlite3
from pathlib import Path

import pytest
from conftest import FakeEmbedder

from quipquill_builder import storyfile
from quipquill_builder.embeddings import unpack_vector
from quipquill_builder.export import ExportError, export_story
from quipquill_builder.models import Entity, Graph
from quipquill_builder.validate import has_errors, validate


def export(graph: Graph, output: Path, embedder: FakeEmbedder, **kwargs):
    kwargs.setdefault("title", "Rune Halls")
    kwargs.setdefault("start", "The Entrance Rune Hall")
    return asyncio.run(export_story(graph, output, embedder, **kwargs))


def test_export_writes_complete_story(
    graph: Graph, fake_embedder: FakeEmbedder, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    output = tmp_path / "rune-halls.qqstory"
    issues = export(graph, output, fake_embedder)

    assert not has_errors(issues)
    assert output.is_file()
    assert not output.with_name(output.name + ".tmp").exists()

    conn = sqlite3.connect(output)
    request.addfinalizer(conn.close)
    meta = storyfile.read_meta(conn)
    assert meta["schema_version"] == "1"
    assert meta["title"] == "Rune Halls"
    assert meta["start_location"] == "loc:entrance-rune-hall"
    assert meta["embed_model"] == "fake/embed"
    assert meta["embed_dim"] == "8"
    assert meta["source_extract_model"] == "fake/model"

    # Tables come from schema/story.sql and are STRICT.
    strict = {row[1]: row[5] for row in conn.execute("PRAGMA table_list") if row[0] == "main"}
    assert strict["entities"] == 1 and strict["relations"] == 1

    # Every entity has a name vector; described entities also have a desc vector.
    count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    names = conn.execute("SELECT COUNT(*) FROM embeddings WHERE kind = 'name'").fetchone()[0]
    descs = conn.execute("SELECT COUNT(*) FROM embeddings WHERE kind = 'desc'").fetchone()[0]
    assert names == count == 7
    assert descs == 6  # Whispering Woods has no description
    blob = conn.execute("SELECT vector FROM embeddings LIMIT 1").fetchone()[0]
    assert len(unpack_vector(blob)) == 8


def test_exported_draft_lacks_only_authored_meta(
    graph: Graph, fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    output = tmp_path / "s.qqstory"
    export(graph, output, fake_embedder)

    with storyfile.connect(output) as conn:
        strict_errors = {i.code for i in validate(conn) if i.level == "error"}
        storyfile.set_meta(conn, {"world_summary": "w", "world_constraints": "c", "style": "s"})
        complete = validate(conn)
    assert strict_errors == {"meta_missing"}
    assert not has_errors(complete)


def test_unreachable_location_aborts_before_embedding(
    graph: Graph, fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    graph.entities.append(Entity("Sunken Vault", "Spot", "Nobody knows the way."))
    output = tmp_path / "broken.qqstory"

    issues = export(graph, output, fake_embedder)

    assert {i.code for i in issues if i.level == "error"} == {"unreachable"}
    assert "loc:sunken-vault" in next(i.message for i in issues if i.code == "unreachable")
    assert fake_embedder.calls == 0
    assert not output.exists()
    assert not output.with_name(output.name + ".tmp").exists()


def test_start_must_be_a_location(
    graph: Graph, fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    with pytest.raises(ExportError, match="not a LOCATION"):
        export(graph, tmp_path / "s.qqstory", fake_embedder, start="copper torch")
    with pytest.raises(ExportError, match="not found"):
        export(graph, tmp_path / "s.qqstory", fake_embedder, start="Atlantis")


def test_default_start_is_first_location(
    graph: Graph, fake_embedder: FakeEmbedder, tmp_path: Path
) -> None:
    output = tmp_path / "s.qqstory"
    export(graph, output, fake_embedder, start=None)

    with storyfile.connect(output) as conn:
        assert storyfile.read_meta(conn)["start_location"] == "loc:entrance-rune-hall"


def test_props_are_written(graph: Graph, fake_embedder: FakeEmbedder, tmp_path: Path) -> None:
    next(e for e in graph.entities if e.name == "rusty key").props = {"fixed": True}
    output = tmp_path / "s.qqstory"
    export(graph, output, fake_embedder)

    with storyfile.connect(output) as conn:
        props = conn.execute("SELECT props FROM entities WHERE id = 'item:rusty-key'").fetchone()
    assert props == ('{"fixed": true}',)
