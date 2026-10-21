"""The story catalogue (ADR 61): list, download, rescan, health."""

import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_auth import bearer, register

from quipquill_cloud.core.config import Settings
from quipquill_cloud.main import create_app

DEMO = Path(__file__).resolve().parents[2] / "stories" / "brass-vault" / "brass-vault.qqstory"
DEMO_ID = "d09514c5-6b96-48a4-ba57-2cc4b4b68f55"


@pytest.fixture
def stories_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "stories"
    folder.mkdir()
    shutil.copy(DEMO, folder / "brass-vault.qqstory")
    return folder


def catalogue_client(settings: Settings, stories_dir: Path) -> TestClient:
    return TestClient(create_app(settings.model_copy(update={"stories_dir": stories_dir})))


def set_meta(path: Path, key: str, value: str) -> None:
    connection = sqlite3.connect(path)
    with connection:
        connection.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    connection.close()


def test_the_catalogue_lists_the_story_and_hands_out_its_file(
    settings: Settings, stories_dir: Path
) -> None:
    with catalogue_client(settings, stories_dir) as client:
        token = register(client).json()["access_token"]
        listing = client.get("/stories", headers=bearer(token))
        assert listing.status_code == 200
        [story] = listing.json()
        assert story["id"] == DEMO_ID
        assert story["title"] == "The Brass Vault"
        assert story["summary"].startswith("The Brass Vault is set in Vellmoor")
        assert story["heroes"] == []
        assert story["size"] == DEMO.stat().st_size
        assert len(story["version"]) == 64

        download = client.get(f"/stories/{DEMO_ID}/file", headers=bearer(token))
        assert download.status_code == 200
        assert download.content == DEMO.read_bytes()
        assert download.headers["etag"] == f'"{story["version"]}"'


def test_the_catalogue_needs_a_signed_in_player(settings: Settings, stories_dir: Path) -> None:
    with catalogue_client(settings, stories_dir) as client:
        assert client.get("/stories").status_code == 401
        assert client.get(f"/stories/{DEMO_ID}/file").status_code == 401


def test_an_unknown_story_is_404(settings: Settings, stories_dir: Path) -> None:
    with catalogue_client(settings, stories_dir) as client:
        token = register(client).json()["access_token"]
        response = client.get("/stories/nope/file", headers=bearer(token))
        assert response.status_code == 404
        assert response.json()["detail"] == "story_not_found"


def test_broken_files_are_skipped(settings: Settings, stories_dir: Path) -> None:
    (stories_dir / "junk.qqstory").write_bytes(b"not a database")
    old = stories_dir / "old-format.qqstory"
    shutil.copy(DEMO, old)
    set_meta(old, "schema_version", "0")
    set_meta(old, "story_id", "old-one")
    with catalogue_client(settings, stories_dir) as client:
        token = register(client).json()["access_token"]
        assert [s["id"] for s in client.get("/stories", headers=bearer(token)).json()] == [DEMO_ID]


def test_tagline_and_heroes_come_from_meta(settings: Settings, stories_dir: Path) -> None:
    path = stories_dir / "brass-vault.qqstory"
    set_meta(path, "tagline", "Steal the Clockheart before noon.")
    set_meta(path, "heroes", '[{"key": "pip", "name": "Pip"}, "not a sheet"]')
    with catalogue_client(settings, stories_dir) as client:
        token = register(client).json()["access_token"]
        [story] = client.get("/stories", headers=bearer(token)).json()
    assert story["tagline"] == "Steal the Clockheart before noon."
    assert story["heroes"] == [{"key": "pip", "name": "Pip"}]


def test_a_new_build_shows_without_a_restart(settings: Settings, stories_dir: Path) -> None:
    with catalogue_client(settings, stories_dir) as client:
        token = register(client).json()["access_token"]
        before = client.get("/stories", headers=bearer(token)).json()[0]["version"]
        set_meta(stories_dir / "brass-vault.qqstory", "built_at", "2099-01-01T00:00:00Z")
        after = client.get("/stories", headers=bearer(token)).json()[0]["version"]
    assert before != after


def test_two_files_of_one_story_offer_the_newer_build(
    settings: Settings, stories_dir: Path
) -> None:
    newer = stories_dir / "nested" / "brass-vault.qqstory"
    newer.parent.mkdir()
    shutil.copy(DEMO, newer)
    set_meta(newer, "built_at", "2099-01-01T00:00:00Z")
    with catalogue_client(settings, stories_dir) as client:
        token = register(client).json()["access_token"]
        [story] = client.get("/stories", headers=bearer(token)).json()
        download = client.get(f"/stories/{DEMO_ID}/file", headers=bearer(token))
    assert download.content == newer.read_bytes()
    assert story["size"] == newer.stat().st_size


def test_health_reports_an_empty_catalogue(
    settings: Settings, tmp_path: Path, stories_dir: Path
) -> None:
    with catalogue_client(settings, tmp_path / "missing") as client:
        response = client.get("/health/stories")
        assert response.status_code == 503
        assert response.json()["detail"] == "no_stories"
    with catalogue_client(settings, stories_dir) as client:
        response = client.get("/health/stories")
        assert response.status_code == 200
        assert response.json()["info"] == {"stories": 1}
