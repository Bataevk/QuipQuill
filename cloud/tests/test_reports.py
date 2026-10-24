"""The game's Report button (POST /reports) and the landing page's waitlist form."""

import json

from conftest import db_run
from fastapi.testclient import TestClient
from test_auth import bearer, register
from test_guest import open_client

from quipquill_cloud.core.config import Settings

REPORT = {
    "message": "The master forgot my lantern.",
    "story_id": "brass-vault",
    "save_id": "save-1",
    "character": {"name": "Wren", "age": "17", "description": "A locksmith's apprentice."},
    "transcript": [
        {"seq": 1, "speaker": "gm", "text": "The vault door is shut.", "ts": "2026-10-24T10:00"},
        {"seq": 2, "speaker": "player", "text": "I light my lantern.", "ts": "2026-10-24T10:01"},
    ],
    "client": {"platform": "ios", "version": "0.1.0"},
}


def test_a_report_is_stored_as_sent(settings: Settings) -> None:
    with open_client(settings) as client:
        token = client.post("/auth/guest").json()["access_token"]
        response = client.post("/reports", json=REPORT, headers=bearer(token))
    assert response.status_code == 201
    (row,) = db_run(
        "SELECT id, message, story_id, character, transcript, client, "
        "user_id IS NOT NULL AS linked "
        "FROM reports"
    )
    assert response.json() == {"id": row["id"]}
    assert (row["message"], row["story_id"], row["linked"]) == (
        REPORT["message"],
        "brass-vault",
        True,
    )
    assert json.loads(row["character"])["name"] == "Wren"
    assert [line["text"] for line in json.loads(row["transcript"])] == [
        "The vault door is shut.",
        "I light my lantern.",
    ]
    assert json.loads(row["client"]) == {"platform": "ios", "version": "0.1.0"}


def test_a_report_checks_its_input(client: TestClient) -> None:
    assert client.post("/reports", json=REPORT).status_code == 401
    token = register(client).json()["access_token"]
    for body in (
        {**REPORT, "message": ""},
        {**REPORT, "transcript": [{"speaker": "gm"}]},
        {"transcript": []},
    ):
        assert client.post("/reports", json=body, headers=bearer(token)).status_code == 422
    huge = {**REPORT, "message": "x" * 3_000_000}
    assert client.post("/reports", json=huge, headers=bearer(token)).status_code == 413
    assert db_run("SELECT count(*) AS n FROM reports")[0]["n"] == 0


def test_the_site_joins_the_waitlist_without_an_account(client: TestClient) -> None:
    body = {"email": "Visitor@Example.com"}
    assert client.post("/waitlist/site", json=body).status_code == 204
    assert client.post("/waitlist/site", json=body).status_code == 204
    assert client.post("/waitlist/site", json={"email": "nope"}).status_code == 422
    rows = db_run("SELECT email, source, user_id FROM waitlist")
    assert [(r["email"], r["source"], r["user_id"]) for r in rows] == [
        ("visitor@example.com", "site", None)
    ]


def test_the_site_form_is_limited_per_address(settings: Settings) -> None:
    with open_client(settings, form_rate_limit_burst=2, form_rate_limit_per_minute=0.001) as client:
        codes = [
            client.post("/waitlist/site", json={"email": f"v{i}@example.com"}).status_code
            for i in range(3)
        ]
    assert codes == [204, 204, 429]
