import json
from pathlib import Path

import pytest

from quipquill_builder import cli, curation
from quipquill_builder.models import Entity, Graph, Relationship

RULES = """
[merge]
"Grand Hall" = ["Ballroom", "the hall"]

[drop]
entities = ["Vellmoor"]

[retype]
"punch card" = "Tool"

[describe]
"Grand Hall" = "  The bank's ballroom tonight.  "

[props]
"wall safe" = { fixed = true }

[[drop_relation]]
source = "Grand Hall"
target = "Cellar"
type = "LEADS_TO"

[[add_relation]]
source = "punch card"
target = "wall safe"
type = "LOCATED_IN"
description = "locked inside"
"""


def graph() -> Graph:
    E, R = Entity, Relationship
    return Graph(
        entities=[
            E("Grand Hall", "Building", ""),
            E("Ballroom", "Building", "Chandeliers."),
            E("the hall", "Spot", "", {"noisy": True}),
            E("Vellmoor", "Country", "A city."),
            E("Cellar", "Spot", "Damp."),
            E("punch card", "Document", "Brass."),
            E("wall safe", "Tool", "Steel."),
        ],
        relationships=[
            R("Ballroom", "Cellar", "LEADS_TO", "down"),
            R("Grand Hall", "Cellar", "INTERACTS_WITH"),
            R("the hall", "Vellmoor", "LOCATED_IN"),
            R("Ballroom", "the hall", "LOCATED_IN"),  # becomes a self-loop after merge
        ],
    )


def rules(tmp_path: Path, text: str = RULES) -> dict:
    path = tmp_path / "curation.toml"
    path.write_text(text, encoding="utf-8")
    return curation.load(path)


def test_apply_all_sections(tmp_path: Path) -> None:
    result = curation.apply(graph(), rules(tmp_path))
    entities = {e.name: e for e in result.entities}

    assert set(entities) == {"Grand Hall", "Cellar", "punch card", "wall safe"}
    assert entities["Grand Hall"].description == "The bank's ballroom tonight."
    assert entities["Grand Hall"].props == {"noisy": True}  # carried over from an alias
    assert entities["punch card"].type == "Tool"
    assert entities["wall safe"].props == {"fixed": True}
    edges = {(r.source, r.type, r.target) for r in result.relationships}
    assert edges == {
        ("Grand Hall", "INTERACTS_WITH", "Cellar"),
        ("punch card", "LOCATED_IN", "wall safe"),
    }
    assert result.meta["curated"] is True


def test_unknown_names_are_errors(tmp_path: Path) -> None:
    bad = rules(tmp_path, '[retype]\n"Grand Hal" = "Spot"\n[drop]\nentities = ["Atlantis"]\n')
    with pytest.raises(curation.CurationError) as error:
        curation.apply(graph(), bad)
    assert "Grand Hal" in str(error.value) and "Atlantis" in str(error.value)


def test_drop_relation_that_does_not_exist_is_an_error(tmp_path: Path) -> None:
    bad = rules(tmp_path, '[[drop_relation]]\nsource = "Cellar"\ntarget = "wall safe"\n')
    with pytest.raises(curation.CurationError, match="drop_relation"):
        curation.apply(graph(), bad)


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(curation.CurationError, match="rename"):
        rules(tmp_path, '[rename]\n"a" = "b"\n')


def test_cli_curate(tmp_path: Path) -> None:
    source = tmp_path / "graph.json"
    source.write_text(json.dumps(graph().to_dict()), encoding="utf-8")
    (tmp_path / "curation.toml").write_text(RULES, encoding="utf-8")
    output = tmp_path / "curated.json"

    assert (
        cli.main(["curate", str(source), str(tmp_path / "curation.toml"), "-o", str(output)]) == 0
    )
    assert len(json.loads(output.read_text())["entities"]) == 4
