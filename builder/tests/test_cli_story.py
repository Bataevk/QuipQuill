import json
from pathlib import Path

import pytest
from conftest import FakeChatClient, FakeEmbedder

from quipquill_builder import cli, storyfile
from quipquill_builder.models import Graph
from quipquill_builder.prompts import META_SYSTEM

DRAFT = {
    "world_summary": "An underground maze of rune halls.",
    "world_constraints": "Low magic. Humans and dwarves only.",
    "style": "Dark fantasy, terse second person.",
}


@pytest.fixture
def graph_file(graph: Graph, tmp_path: Path) -> Path:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    return path


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> FakeChatClient:
    client = FakeChatClient({"Entities:": json.dumps(DRAFT)})
    monkeypatch.setattr(cli, "OpenRouterEmbedder", lambda settings, model: FakeEmbedder())
    monkeypatch.setattr(cli, "OpenRouterChatClient", lambda settings, model: client)
    return client


def test_export_meta_validate_flow(
    graph_file: Path, tmp_path: Path, patched: FakeChatClient, capsys: pytest.CaptureFixture
) -> None:
    story = tmp_path / "rune-halls.qqstory"

    assert (
        cli.main(["export", str(graph_file), "-o", str(story), "--start", "loc:entrance-rune-hall"])
        == 0
    )
    assert cli.main(["validate", str(story)]) == 1  # world texts not written yet
    assert cli.main(["validate", str(story), "--draft"]) == 0

    assert cli.main(["meta", str(story)]) == 0
    assert patched.calls[-1][0] == META_SYSTEM
    assert "[LOCATION] The Entrance Rune Hall" in patched.calls[-1][1]
    assert cli.main(["validate", str(story)]) == 0

    assert cli.main(["meta", str(story), "--set", "style=Cosy mystery."]) == 0
    with storyfile.connect(story) as conn:
        meta = storyfile.read_meta(conn)
    assert meta["style"] == "Cosy mystery."
    assert meta["world_summary"] == DRAFT["world_summary"]
    assert meta["title"] == "Rune Halls"

    capsys.readouterr()
    assert cli.main(["meta", str(story), "--show"]) == 0
    assert "start_location: loc:entrance-rune-hall" in capsys.readouterr().out


def test_meta_does_not_redraft_without_force(
    graph_file: Path, tmp_path: Path, patched: FakeChatClient
) -> None:
    story = tmp_path / "s.qqstory"
    cli.main(["export", str(graph_file), "-o", str(story)])
    cli.main(["meta", str(story)])
    calls = len(patched.calls)

    assert cli.main(["meta", str(story)]) == 0
    assert len(patched.calls) == calls
    assert cli.main(["meta", str(story), "--force"]) == 0
    assert len(patched.calls) == calls + 1


def test_export_rejects_broken_graph(graph: Graph, tmp_path: Path, patched: FakeChatClient) -> None:
    graph.relationships = [r for r in graph.relationships if r.type != "LEADS_TO"]
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")
    story = tmp_path / "s.qqstory"

    assert cli.main(["export", str(path), "-o", str(story)]) == 1
    assert not story.exists()


def test_meta_set_requires_key_value(
    graph_file: Path, tmp_path: Path, patched: FakeChatClient
) -> None:
    story = tmp_path / "s.qqstory"
    cli.main(["export", str(graph_file), "-o", str(story)])
    assert cli.main(["meta", str(story), "--set", "nonsense"]) == 1
