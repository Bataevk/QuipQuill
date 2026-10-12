import json

import pytest

from quipquill_builder.text import normalize_name, parse_json_payload, split_text


def test_short_text_is_one_chunk() -> None:
    assert split_text("One paragraph.\n\nAnother one.", 100, 10) == [
        "One paragraph.\n\nAnother one."
    ]


def test_chunks_respect_size_and_share_overlap() -> None:
    paragraphs = [f"Paragraph {i} " + "x" * 40 for i in range(10)]
    chunks = split_text("\n\n".join(paragraphs), chunk_size=150, overlap=60)

    assert len(chunks) > 1
    assert all(len(c) <= 150 for c in chunks)
    # The last paragraph of a chunk reappears at the start of the next one.
    for current, following in zip(chunks, chunks[1:], strict=False):
        assert following.startswith(current.split("\n\n")[-1])
    # Nothing is lost.
    assert all(any(p in c for c in chunks) for p in paragraphs)


def test_long_paragraph_is_split_by_sentences() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(20))
    chunks = split_text(text, chunk_size=80, overlap=0)

    assert all(len(c) <= 80 for c in chunks)
    assert chunks[0].startswith("Sentence number 0")


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_params(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("text", size, overlap)


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("The Entrance Rune Hall", "entrance rune hall"),
        ("Torchlight Corridor.", "torchlight corridor"),
        ("half-buried  dagger<|im_end|>", "half buried dagger"),
    ],
)
def test_normalize_name(name: str, key: str) -> None:
    assert normalize_name(name) == key


def test_parse_json_payload_strips_fences_and_think_blocks() -> None:
    raw = '<think>hmm</think>\n```json\n{"entities": []}\n```'
    assert parse_json_payload(raw) == {"entities": []}


def test_parse_json_payload_keeps_smart_quotes_inside_strings() -> None:
    raw = json.dumps({"description": "reads: “Beware”"}, ensure_ascii=False)
    assert parse_json_payload(raw) == {"description": "reads: “Beware”"}


def test_parse_json_payload_repairs_smart_quote_delimiters() -> None:
    raw = "{“entities”: []}"
    assert parse_json_payload(raw) == {"entities": []}


def test_parse_json_payload_finds_object_in_prose() -> None:
    assert parse_json_payload('Here you go: {"a": 1} Hope it helps.') == {"a": 1}


def test_parse_json_payload_raises_on_garbage() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_payload("not json at all {")
