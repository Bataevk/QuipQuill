"""graph.json → .qqstory: story rows, embeddings, meta, validation."""

import logging
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from quipquill_builder import storyfile
from quipquill_builder.embeddings import Embedder, pack_vector
from quipquill_builder.models import Graph
from quipquill_builder.story import StoryData, build_story
from quipquill_builder.validate import Issue, has_errors, validate

logger = logging.getLogger(__name__)


class ExportError(RuntimeError):
    pass


def builder_version() -> str:
    try:
        return version("quipquill-builder")
    except PackageNotFoundError:
        return "0.0.0"


def pick_start(story: StoryData, start: str | None) -> str:
    if start:
        entity = story.find(start)
        if entity is None:
            raise ExportError(f"Start location {start!r} not found in the graph")
        if entity.type != "LOCATION":
            raise ExportError(f"Start location {start!r} is a {entity.type}, not a LOCATION")
        return entity.id
    first = next((e for e in story.entities if e.type == "LOCATION"), None)
    if first is None:
        raise ExportError("The graph has no locations to start in")
    logger.warning("No --start given; using the first location: %s (%s)", first.name, first.id)
    return first.id


def embedding_inputs(story: StoryData) -> list[tuple[str, str, str]]:
    """(entity_id, kind, text) for every vector to compute."""
    inputs = []
    for entity in story.entities:
        inputs.append((entity.id, "name", entity.name))
        if entity.description.strip():
            inputs.append((entity.id, "desc", f"{entity.name}: {entity.description}"))
    return inputs


async def export_story(
    graph: Graph,
    output: Path,
    embedder: Embedder,
    *,
    title: str,
    start: str | None = None,
    language: str = "en",
    story_id: str | None = None,
    authored: dict[str, str] | None = None,
) -> list[Issue]:
    """Build the story file. On validation errors nothing is written and issues are returned.

    `story_id` and `authored` (world_summary, world_constraints, style, luck) come from meta.toml
    so a rebuild keeps the story's identity and texts (ADR 24); without them the id is new.
    """
    story = build_story(graph)
    meta = {
        "schema_version": storyfile.SCHEMA_VERSION,
        "story_id": story_id or str(uuid.uuid4()),
        "title": title,
        "language": language,
        "start_location": pick_start(story, start),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "builder_version": builder_version(),
        "source_extract_model": str(graph.meta.get("extract_model", "")),
    }
    kept = storyfile.AUTHORED_META + storyfile.OPTIONAL_META
    meta.update({k: v for k, v in (authored or {}).items() if k in kept})

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    try:
        storyfile.create(tmp, story, meta)
        # Structure first: no point paying for embeddings of a broken graph.
        with storyfile.connect(tmp) as conn:
            issues = validate(conn, check_embeddings=False, require_authored_meta=False)
        if has_errors(issues):
            return issues

        inputs = embedding_inputs(story)
        logger.info("Embedding %d texts with %s", len(inputs), embedder.model)
        vectors = await embedder.embed([text for _, _, text in inputs])
        with storyfile.connect(tmp) as conn:
            conn.executemany(
                "INSERT INTO embeddings (entity_id, kind, vector) VALUES (?, ?, ?)",
                [
                    (entity_id, kind, pack_vector(vector))
                    for (entity_id, kind, _), vector in zip(inputs, vectors, strict=True)
                ],
            )
            storyfile.set_meta(
                conn, {"embed_model": embedder.model, "embed_dim": str(len(vectors[0]))}
            )
            issues = validate(conn, require_authored_meta=False)
        if has_errors(issues):
            return issues

        tmp.replace(output)
        return issues
    finally:
        tmp.unlink(missing_ok=True)
