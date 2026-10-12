"""Text helpers: loading lore files, chunking, name normalization, LLM JSON cleanup."""

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = (".md", ".txt")

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ARTICLE_RE = re.compile(r"\b(?:a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True)
class Document:
    path: Path
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Load `.md` / `.txt` files from a directory (sorted, non-recursive)."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Lore directory not found: {directory}")
    documents = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                documents.append(Document(path=path, text=text))
    return documents


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of at most ~`chunk_size` characters on paragraph boundaries.

    Consecutive chunks share trailing paragraphs worth up to `overlap` characters so that
    facts spanning a boundary are seen whole at least once. A single paragraph longer than
    `chunk_size` is split on sentence boundaries, and as a last resort by characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if paragraph:
            pieces.extend(_split_long(paragraph, chunk_size))

    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and len("\n\n".join([*current, piece])) > chunk_size:
            chunks.append("\n\n".join(current))
            current = _tail_within(current, overlap)
            while current and len("\n\n".join([*current, piece])) > chunk_size:
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_long(paragraph: str, chunk_size: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]
    parts: list[str] = []
    buffer = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        candidate = f"{buffer} {sentence}".strip()
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            parts.append(buffer)
        while len(sentence) > chunk_size:
            parts.append(sentence[:chunk_size])
            sentence = sentence[chunk_size:]
        buffer = sentence
    if buffer:
        parts.append(buffer)
    return parts


def _tail_within(pieces: list[str], budget: int) -> list[str]:
    tail: list[str] = []
    size = 0
    for piece in reversed(pieces):
        size += len(piece) + 2
        if size > budget:
            break
        tail.insert(0, piece)
    return tail


def remove_special_tokens(text: str) -> str:
    """Remove chat-template tokens like `<|im_end|>` that small models sometimes echo."""
    return _SPECIAL_TOKEN_RE.sub("", text or "")


def normalize_name(name: str) -> str:
    """Comparison key for entity names: lowercase, no punctuation, articles or extra spaces.

    Used only for matching/merging; the display name keeps its original form.
    """
    text = remove_special_tokens(name).replace("-", " ").lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_json_payload(raw: str) -> Any:
    """Parse a JSON object from an LLM reply, tolerating fences, <think> blocks and smart quotes.

    Smart quotes are only replaced as a fallback: inside valid JSON strings they are content
    (e.g. a sign that reads “Beware”), and replacing them would break the string.
    """
    text = _strip_wrapping(_THINK_RE.sub("", raw or "").strip())
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = text.replace("“", '"').replace("”", '"')
        return json.loads(fixed)


def _strip_wrapping(text: str) -> str:
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text
