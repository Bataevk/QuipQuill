"""Prompts live in the cloud (ADR 57), not in the app: the client names a template in a message
part — `{"type": "prompt", "name": "gm", "vars": {"STORY_STYLE": "…"}}` — and the worker puts the
template's text there, keeping the part's `cache_control`. Templates are files `<name>.md` in
PROMPTS_DIR: in production written from the CI variables `PROMPT_<NAME>`
(docs/notes/deploy.md), in development the repository's `prompts/`.

Rendering is deterministic (the same template and vars give the same bytes), so a prefix the
provider cached stays cached; a changed template is a new prefix once. A file is read again when
it changes on disk, without a restart.
"""

import re
from pathlib import Path
from typing import Any

# Every template the app asks for; /health/prompts reports the missing ones.
PROMPT_NAMES = ("gm", "actor", "guard", "generator", "character", "summary", "note_entity")

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_VAR = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


class PromptError(ValueError):
    """The request names a template that does not exist or gives it the wrong variables."""


class PromptStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        # name → (mtime_ns, size, text): a file is read again only when it changed.
        self._cache: dict[str, tuple[int, int, str]] = {}

    def template(self, name: str) -> str:
        if not _NAME.match(name):
            raise PromptError(f"bad prompt name {name!r}")
        path = self._directory / f"{name}.md"
        try:
            stat = path.stat()
        except OSError as error:
            raise PromptError(f"no prompt {name!r}") from error
        cached = self._cache.get(name)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]
        text = path.read_text(encoding="utf-8").rstrip()
        self._cache[name] = (stat.st_mtime_ns, stat.st_size, text)
        return text

    def render(self, name: str, variables: dict[str, Any]) -> str:
        """The template with each `{{VAR}}` replaced in one pass: every var it uses, no others."""
        template = self.template(name)
        wanted = set(_VAR.findall(template))
        given = set(variables)
        if wanted != given:
            raise PromptError(
                f"prompt {name!r} takes {sorted(wanted)}, got {sorted(given)}",
            )
        if not all(isinstance(value, str) for value in variables.values()):
            raise PromptError(f"prompt {name!r}: variables must be strings")
        return _VAR.sub(lambda match: variables[match.group(1)], template)

    def expand(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The messages with every prompt part turned into a text part (new lists, input intact)."""
        return [_expand_message(self, message) for message in messages]

    def missing(self) -> list[str]:
        return [name for name in PROMPT_NAMES if not (self._directory / f"{name}.md").is_file()]


def _expand_message(store: PromptStore, message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list) or not any(_is_prompt(part) for part in content):
        return message
    return {**message, "content": [_expand_part(store, part) for part in content]}


def _is_prompt(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "prompt"


def _expand_part(store: PromptStore, part: Any) -> Any:
    if not _is_prompt(part):
        return part
    extra = set(part) - {"type", "name", "vars", "cache_control"}
    if extra:
        raise PromptError(f"prompt part has unknown fields {sorted(extra)}")
    variables = part.get("vars") or {}
    if not isinstance(variables, dict):
        raise PromptError("prompt vars must be an object")
    text = store.render(str(part.get("name", "")), variables)
    cache = part.get("cache_control")
    return {"type": "text", "text": text, **({"cache_control": cache} if cache else {})}


# A question of a decision template (ADR 70): `## name (type): instructions`, then for `choice`
# the options as `- KEY: what it means`, for `score` the levels as `- what it means` (lowest first).
_QUESTION = re.compile(r"^## ([a-z][a-z0-9_]*) \((choice|score|noul)\): (.+)$")
_OPTION = re.compile(r"^- ([A-Z][A-Z0-9_]*): (.+)$")
_LEVEL = re.compile(r"^- (.+)$")


def split_decision(text: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """A rendered decision template → the text that opens the model's state, and its questions
    in the Decisions API's form (ADR 70). The questions are prompt text too, so they live in the
    template, not in the app."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if _QUESTION.match(line)]
    if not starts:
        raise PromptError("a decision template needs at least one question")
    questions: dict[str, dict[str, Any]] = {}
    for n, start in enumerate(starts):
        name, kind, instructions = _QUESTION.match(lines[start]).groups()  # type: ignore[union-attr]
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        body = [line for line in lines[start + 1 : end] if line.strip()]
        question: dict[str, Any] = {"type": kind, "instructions": instructions.strip()}
        if kind == "choice":
            options = [_OPTION.match(line) for line in body]
            if not options or not all(options):
                raise PromptError(f"choice {name!r}: every line must be '- KEY: meaning'")
            question["criteria"] = {m.group(1): m.group(2).strip() for m in options if m}
        elif kind == "score":
            levels = [_LEVEL.match(line) for line in body]
            if len(levels) < 2 or not all(levels):
                raise PromptError(f"score {name!r}: at least two '- level' lines")
            question["criteria"] = [m.group(1).strip() for m in levels if m]
        elif body:
            raise PromptError(f"noul {name!r} takes no options")
        questions[name] = question
    return "\n".join(lines[: starts[0]]).strip(), questions
