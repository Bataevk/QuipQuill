"""Lore → story graph extraction (port of the legacy `GraphExtractor` from `main`).

Pipeline: split documents into chunks → extract entities/relationships per chunk with an LLM
→ merge near-duplicate entities (fuzzy name match) → summarize merged descriptions → resolve
relationship endpoints to canonical entities.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz, process

from quipquill_builder import prompts
from quipquill_builder.llm import BulkChatClient, ChatClient, Prompt
from quipquill_builder.models import Entity, Graph, Relationship
from quipquill_builder.text import (
    Document,
    normalize_name,
    parse_json_payload,
    remove_special_tokens,
    split_text,
)

logger = logging.getLogger(__name__)

DEFAULT_TYPE = "OTHER"
DESCRIPTION_SEPARATOR = "\n"


class ExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractionParams:
    chunk_size: int = 2000
    chunk_overlap: int = 200
    merge_threshold: float = 0.85
    max_concurrency: int = 4


@dataclass
class _MergedEntity:
    name: str
    type: str
    descriptions: list[str] = field(default_factory=list)

    def add_description(self, description: str) -> None:
        if description and description not in self.descriptions:
            self.descriptions.append(description)


class GraphExtractor:
    def __init__(
        self,
        extractor: ChatClient,
        summarizer: ChatClient | None = None,
        params: ExtractionParams | None = None,
    ) -> None:
        self._extractor = extractor
        self._summarizer = summarizer or extractor
        self._params = params or ExtractionParams()

    async def extract(self, documents: list[Document]) -> Graph:
        chunks = [
            chunk
            for document in documents
            for chunk in split_text(
                document.text, self._params.chunk_size, self._params.chunk_overlap
            )
        ]
        if not chunks:
            raise ExtractionError("No text to process")
        logger.info("Extracting from %d chunks of %d documents", len(chunks), len(documents))

        replies = await self._complete_all(
            self._extractor,
            [(prompts.EXTRACTION_SYSTEM, prompts.EXTRACTION_USER.format(text=c)) for c in chunks],
        )
        valid = [r for r in map(_parse_chunk_reply, replies) if r is not None]
        if not valid:
            raise ExtractionError("Every chunk failed to extract; see log for details")

        raw_entities = [e for r in valid for e in _coerce_entities(r)]
        raw_relationships = [rel for r in valid for rel in _coerce_relationships(r)]
        logger.info(
            "Raw extraction: %d entities, %d relationships",
            len(raw_entities),
            len(raw_relationships),
        )

        merged = _merge_entities(raw_entities, self._params.merge_threshold)
        descriptions = await self._summarize_all(merged)
        entities = [
            Entity(name=m.name, type=m.type, description=d)
            for m, d in zip(merged, descriptions, strict=True)
        ]
        entities, relationships = _resolve_relationships(
            entities, raw_relationships, self._params.merge_threshold
        )
        logger.info("Final graph: %d entities, %d relationships", len(entities), len(relationships))

        return Graph(
            entities=entities,
            relationships=relationships,
            meta={
                "extract_model": self._extractor.model,
                "summarize_model": self._summarizer.model,
                "documents": [d.path.name for d in documents],
                "chunks": len(chunks),
                "failed_chunks": len(chunks) - len(valid),
            },
        )

    async def _complete_all(
        self, client: ChatClient, requests: list[Prompt]
    ) -> list[str | Exception]:
        """One provider batch when the client supports it, otherwise bounded parallel calls."""
        if isinstance(client, BulkChatClient):
            return await client.complete_many(requests)
        semaphore = asyncio.Semaphore(self._params.max_concurrency)

        async def run(system: str, user: str) -> str | Exception:
            async with semaphore:
                try:
                    return await client.complete(system, user)
                except Exception as error:
                    logger.exception("LLM call failed")
                    return error

        return await asyncio.gather(*(run(system, user) for system, user in requests))

    async def _summarize_all(self, merged: list[_MergedEntity]) -> list[str]:
        """Merged entities with several descriptions get one summary each; others pass through."""
        result = [DESCRIPTION_SEPARATOR.join(m.descriptions) for m in merged]
        todo = [i for i, m in enumerate(merged) if len(m.descriptions) > 1]
        if not todo:
            return result
        logger.info("Summarizing descriptions of %d merged entities", len(todo))
        replies = await self._complete_all(
            self._summarizer,
            [
                (prompts.SUMMARIZATION_SYSTEM, prompts.SUMMARIZATION_USER.format(text=result[i]))
                for i in todo
            ],
        )
        for i, reply in zip(todo, replies, strict=True):
            if isinstance(reply, Exception):
                logger.error("Summary failed for '%s'; keeping joined text", merged[i].name)
                continue
            result[i] = remove_special_tokens(reply).strip() or result[i]
        return result


def _parse_chunk_reply(reply: str | Exception) -> dict[str, Any] | None:
    if isinstance(reply, Exception):
        logger.error("Chunk skipped: extraction call failed (%s)", reply)
        return None
    try:
        data = parse_json_payload(reply)
    except json.JSONDecodeError as error:
        logger.error("Chunk skipped: reply is not valid JSON (%s)", error)
        return None
    if not isinstance(data, dict) or "entities" not in data:
        logger.warning("Chunk skipped: reply has no 'entities' key")
        return None
    return data


def _field(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return remove_special_tokens(value).strip()
    return ""


def _coerce_entities(result: dict[str, Any]) -> list[Entity]:
    fields = prompts.ENTITY_FIELDS
    entities = []
    for item in result.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = _field(item, fields["name"])
        if not name:
            continue
        entities.append(
            Entity(
                name=name,
                type=_field(item, fields["type"]) or DEFAULT_TYPE,
                description=_field(item, fields["description"]),
            )
        )
    return entities


def _coerce_relationships(result: dict[str, Any]) -> list[Relationship]:
    fields = prompts.RELATIONSHIP_FIELDS
    relationships = []
    for item in result.get("relationships") or []:
        if not isinstance(item, dict):
            continue
        # Accept both the prompt's field names and plain ones some models fall back to.
        source = _field(item, fields["source"], "source")
        target = _field(item, fields["target"], "target")
        if not source or not target:
            continue
        relationships.append(
            Relationship(
                source=source,
                target=target,
                type=(_field(item, fields["type"], "type") or DEFAULT_TYPE).upper(),
                description=_field(item, fields["description"], "description"),
            )
        )
    return relationships


class _NameIndex:
    """Canonical entity names looked up by normalized key, exactly or fuzzily."""

    def __init__(self, threshold: float) -> None:
        self._cutoff = threshold * 100
        self._by_key: dict[str, str] = {}

    def add(self, name: str) -> None:
        self._by_key.setdefault(normalize_name(name) or name.lower(), name)

    def find(self, name: str) -> str | None:
        key = normalize_name(name) or name.lower()
        if key in self._by_key:
            return self._by_key[key]
        match = process.extractOne(
            key, self._by_key.keys(), scorer=fuzz.ratio, score_cutoff=self._cutoff
        )
        return self._by_key[match[0]] if match else None


def _merge_entities(raw: list[Entity], threshold: float) -> list[_MergedEntity]:
    index = _NameIndex(threshold)
    merged: dict[str, _MergedEntity] = {}
    for entity in raw:
        canonical = index.find(entity.name)
        if canonical is None:
            index.add(entity.name)
            merged[entity.name] = _MergedEntity(name=entity.name, type=entity.type)
            canonical = entity.name
        elif canonical != entity.name:
            logger.debug("Merged '%s' into '%s'", entity.name, canonical)
        target = merged[canonical]
        if target.type == DEFAULT_TYPE and entity.type != DEFAULT_TYPE:
            target.type = entity.type
        target.add_description(entity.description)
    logger.info("Merged %d raw entities into %d", len(raw), len(merged))
    return list(merged.values())


def _resolve_relationships(
    entities: list[Entity], raw: list[Relationship], threshold: float
) -> tuple[list[Entity], list[Relationship]]:
    """Map relationship endpoints to canonical entities.

    An endpoint that matches no entity becomes a new entity with an empty description
    (legacy behavior: the text mentioned it, so the world should know it exists).
    """
    entities = list(entities)
    index = _NameIndex(threshold)
    for entity in entities:
        index.add(entity.name)

    def resolve(name: str) -> str:
        canonical = index.find(name)
        if canonical is None:
            logger.info("Adding entity '%s' referenced only by a relationship", name)
            entities.append(Entity(name=name, type=DEFAULT_TYPE, description=""))
            index.add(name)
            canonical = name
        return canonical

    relationships: dict[tuple[str, str, str], Relationship] = {}
    for rel in raw:
        source, target = resolve(rel.source), resolve(rel.target)
        if source == target:
            continue
        key = (source, target, rel.type)
        existing = relationships.get(key)
        if existing is None:
            relationships[key] = Relationship(source, target, rel.type, rel.description)
        elif not existing.description and rel.description:
            existing.description = rel.description
    return entities, list(relationships.values())
