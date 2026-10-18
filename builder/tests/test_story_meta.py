import json
from pathlib import Path

import pytest
from conftest import FakeChatClient, FakeEmbedder

from quipquill_builder import cli, story_meta, storyfile
from quipquill_builder.models import Graph
from quipquill_builder.validate import validate

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


@pytest.fixture(autouse=True)
def patched(monkeypatch: pytest.MonkeyPatch) -> FakeChatClient:
    client = FakeChatClient({"Entities:": json.dumps(DRAFT)})
    monkeypatch.setattr(cli, "OpenRouterBatchEmbedder", lambda settings, model: FakeEmbedder())
    monkeypatch.setattr(cli, "OpenRouterChatClient", lambda settings, model: client)
    return client


def story_meta_of(path: Path) -> dict[str, str]:
    with storyfile.connect(path) as conn:
        return storyfile.read_meta(conn)


def test_round_trip_keeps_quotes_backslashes_and_lines(tmp_path: Path) -> None:
    values = {
        "story_id": "id-1",
        "title": 'The "Brass" Vault',
        "language": "en",
        "world_summary": 'Line one.\nLine "two" with a \\ backslash.\n\n"""Triple""" quotes.',
        "style": 'Ends with a quote"',
    }
    path = tmp_path / "meta.toml"
    story_meta.save(path, values)
    assert story_meta.load(path) == values


def test_load_rejects_unknown_keys_and_missing_id(tmp_path: Path) -> None:
    path = tmp_path / "meta.toml"
    path.write_text('story_id = "x"\nstart = "hall"\n', encoding="utf-8")
    with pytest.raises(story_meta.StoryMetaError, match="unknown keys start"):
        story_meta.load(path)
    path.write_text('title = "T"\n', encoding="utf-8")
    with pytest.raises(story_meta.StoryMetaError, match="story_id is required"):
        story_meta.load(path)


def test_rebuild_keeps_story_id_title_and_world_texts(graph_file: Path, tmp_path: Path) -> None:
    story = tmp_path / "rune-halls.qqstory"
    toml = tmp_path / "meta.toml"
    export = ["export", str(graph_file), "-o", str(story), "--meta", str(toml)]

    assert cli.main([*export, "--title", "Rune Halls"]) == 0
    created = story_meta.load(toml)
    assert created["title"] == "Rune Halls"
    assert story_meta_of(story)["story_id"] == created["story_id"]

    assert cli.main(["meta", str(story), "--meta", str(toml)]) == 0  # LLM draft
    assert cli.main(["meta", str(story), "--meta", str(toml), "--set", "style=Cosy."]) == 0
    assert story_meta.load(toml)["style"] == "Cosy."

    assert cli.main([*export, "--title", "Ignored"]) == 0  # rebuild
    rebuilt = story_meta_of(story)
    assert rebuilt["story_id"] == created["story_id"]
    assert rebuilt["title"] == "Rune Halls"
    assert rebuilt["world_summary"] == DRAFT["world_summary"]
    assert rebuilt["style"] == "Cosy."
    assert cli.main(["validate", str(story)]) == 0  # authored meta survived


def test_meta_bootstraps_toml_from_an_existing_story(graph_file: Path, tmp_path: Path) -> None:
    story = tmp_path / "s.qqstory"
    cli.main(["export", str(graph_file), "-o", str(story), "--title", "Old Story"])
    cli.main(["meta", str(story), "--set", "world_summary=Kept."])
    toml = tmp_path / "stories" / "s" / "meta.toml"

    assert cli.main(["meta", str(story), "--meta", str(toml), "--show"]) == 0
    data = story_meta.load(toml)
    assert data["story_id"] == story_meta_of(story)["story_id"]
    assert (data["title"], data["world_summary"]) == ("Old Story", "Kept.")
    assert "style" not in data  # empty texts are not copied


def test_luck_goes_from_meta_toml_into_the_story(graph_file: Path, tmp_path: Path) -> None:
    story = tmp_path / "rune-halls.qqstory"
    toml = tmp_path / "meta.toml"
    export = ["export", str(graph_file), "-o", str(story), "--meta", str(toml), "--title", "Halls"]
    assert cli.main(export) == 0
    assert "luck" not in story_meta_of(story)  # a fair world unless the author says

    assert cli.main(["meta", str(story), "--meta", str(toml), "--set", "luck=0.2"]) == 0
    assert story_meta.load(toml)["luck"] == "0.2"
    assert cli.main(export) == 0  # rebuild keeps it
    assert story_meta_of(story)["luck"] == "0.2"

    assert cli.main(["meta", str(story), "--set", "luck=lucky"]) == 0
    with storyfile.connect(story) as conn:
        issues = validate(conn, check_embeddings=False, require_authored_meta=False)
    assert "meta_luck" in [i.code for i in issues]
