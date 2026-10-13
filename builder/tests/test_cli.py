import json
from pathlib import Path

import pytest
from conftest import FakeChatClient

from quipquill_builder import cli


def test_extract_writes_graph_json(
    monkeypatch: pytest.MonkeyPatch,
    fake_client: FakeChatClient,
    lore_dir: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli, "OpenRouterBatchClient", lambda settings, model, **kw: fake_client)
    output = tmp_path / "out" / "graph.json"

    code = cli.main(["extract", str(lore_dir), "-o", str(output), "--chunk-size", "600"])

    assert code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    names = {e["name"] for e in data["entities"]}
    assert {"The Entrance Rune Hall", "Torchlight Corridor", "Whispering Woods"} <= names
    assert any(r["type"] == "LEADS_TO" for r in data["relationships"])


def test_extract_without_api_key_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, lore_dir: Path, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no builder/.env here

    code = cli.main(["extract", str(lore_dir), "-o", str(tmp_path / "graph.json")])

    assert code == 1
    assert not (tmp_path / "graph.json").exists()


def test_missing_input_dir_fails_cleanly(tmp_path: Path) -> None:
    assert cli.main(["extract", str(tmp_path / "nope")]) == 1
