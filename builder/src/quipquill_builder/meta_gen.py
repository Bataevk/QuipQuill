"""Drafting the authored meta (world summary, constraints, style) with one LLM call."""

import sqlite3

from quipquill_builder.llm import ChatClient
from quipquill_builder.prompts import META_SYSTEM
from quipquill_builder.storyfile import AUTHORED_META
from quipquill_builder.text import parse_json_payload

MAX_DESCRIPTION_CHARS = 300
TYPE_ORDER = ("LOCATION", "NPC", "CREATURE", "ORGANIZATION", "ITEM", "DOCUMENT", "OTHER")


def describe_world(conn: sqlite3.Connection, title: str) -> str:
    rows = conn.execute("SELECT type, name, description FROM entities").fetchall()
    rows.sort(key=lambda r: (TYPE_ORDER.index(r[0]) if r[0] in TYPE_ORDER else 99, r[1]))
    lines = [f"Story title: {title}", "", "Entities:"]
    for etype, name, description in rows:
        text = description.strip().replace("\n", " ")
        if len(text) > MAX_DESCRIPTION_CHARS:
            text = text[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0] + "…"
        lines.append(f"- [{etype}] {name}: {text}" if text else f"- [{etype}] {name}")
    return "\n".join(lines)


async def draft_meta(client: ChatClient, conn: sqlite3.Connection, title: str) -> dict[str, str]:
    reply = await client.complete(META_SYSTEM, describe_world(conn, title))
    data = parse_json_payload(reply)
    if not isinstance(data, dict):
        raise ValueError("Meta draft is not a JSON object")
    draft = {key: str(data.get(key, "")).strip() for key in AUTHORED_META}
    missing = [key for key, value in draft.items() if not value]
    if missing:
        raise ValueError(f"Meta draft is missing: {', '.join(missing)}")
    return draft
