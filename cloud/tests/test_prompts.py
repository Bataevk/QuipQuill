"""Prompt templates in the cloud (ADR 57): the client names one, the worker fills it in."""

import os
from pathlib import Path

import pytest

from quipquill_cloud.core.config import REPO_PROMPTS_DIR
from quipquill_cloud.llm.prompts import PROMPT_NAMES, PromptError, PromptStore


@pytest.fixture
def store(tmp_path: Path) -> PromptStore:
    (tmp_path / "gm.md").write_text("Rules.\n\n# Style\n{{STORY_STYLE}}\n")
    (tmp_path / "plain.md").write_text("No variables.\n")
    return PromptStore(tmp_path)


def test_renders_every_variable_in_one_pass(store: PromptStore) -> None:
    # A value that looks like a placeholder stays as it is.
    assert store.render("gm", {"STORY_STYLE": "Noir {{X}}"}) == "Rules.\n\n# Style\nNoir {{X}}"
    assert store.render("plain", {}) == "No variables."


@pytest.mark.parametrize(
    ("name", "variables"),
    [
        ("gm", {}),  # missing
        ("gm", {"STORY_STYLE": "x", "EXTRA": "y"}),  # unknown
        ("gm", {"STORY_STYLE": 3}),  # not a string
        ("nope", {}),
        ("../secret", {}),
    ],
)
def test_refuses_wrong_names_and_variables(store: PromptStore, name: str, variables: dict) -> None:
    with pytest.raises(PromptError):
        store.render(name, variables)


def test_expands_prompt_parts_and_keeps_their_cache_breakpoint(store: PromptStore) -> None:
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "prompt",
                    "name": "gm",
                    "vars": {"STORY_STYLE": "Noir"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
                {"type": "text", "text": "Scene."},
            ],
        },
        {"role": "user", "content": "Hi."},
    ]
    expanded = store.expand(messages)
    assert expanded[0]["content"] == [
        {
            "type": "text",
            "text": "Rules.\n\n# Style\nNoir",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
        {"type": "text", "text": "Scene."},
    ]
    assert expanded[1] is messages[1]
    assert messages[0]["content"][0]["type"] == "prompt"  # the request itself is untouched


def test_reads_a_changed_file_again(store: PromptStore, tmp_path: Path) -> None:
    assert store.render("plain", {}) == "No variables."
    path = tmp_path / "plain.md"
    path.write_text("Changed, and longer.\n")
    os.utime(path, ns=(1, 1))
    assert store.render("plain", {}) == "Changed, and longer."


def test_the_repository_has_every_prompt() -> None:
    assert PromptStore(REPO_PROMPTS_DIR).missing() == []
    assert set(PROMPT_NAMES) == {p.stem for p in REPO_PROMPTS_DIR.glob("*.md")}
