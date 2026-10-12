"""Command line entry point: `uv run builder <command>`."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from quipquill_builder.config import Settings
from quipquill_builder.extractor import ExtractionError, ExtractionParams, GraphExtractor
from quipquill_builder.llm import OpenRouterChatClient
from quipquill_builder.text import load_documents


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="builder", description="QuipQuill story builder")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="lore files → graph.json")
    extract.add_argument("input_dir", type=Path, help="directory with .md/.txt lore files")
    extract.add_argument("-o", "--output", type=Path, default=Path("graph.json"))
    extract.add_argument("--model", default=settings.extract_model)
    extract.add_argument("--summarize-model", default=settings.summarize_model)
    extract.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    extract.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    extract.add_argument("--merge-threshold", type=float, default=settings.merge_threshold)
    extract.add_argument("--concurrency", type=int, default=settings.max_concurrency)
    return parser


def run_extract(args: argparse.Namespace, settings: Settings) -> int:
    documents = load_documents(args.input_dir)
    if not documents:
        logging.error("No .md/.txt files in %s", args.input_dir)
        return 1
    extractor_client = OpenRouterChatClient(settings, args.model)
    summarizer_client = (
        OpenRouterChatClient(settings, args.summarize_model)
        if args.summarize_model and args.summarize_model != args.model
        else extractor_client
    )
    extractor = GraphExtractor(
        extractor_client,
        summarizer_client,
        ExtractionParams(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            merge_threshold=args.merge_threshold,
            max_concurrency=args.concurrency,
        ),
    )
    try:
        graph = asyncio.run(extractor.extract(documents))
    except ExtractionError as error:
        logging.error("%s", error)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(graph.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{len(graph.entities)} entities, {len(graph.relationships)} relationships "
        f"({graph.meta['failed_chunks']}/{graph.meta['chunks']} chunks failed) → {args.output}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    args = build_parser(settings).parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # The OpenAI SDK logs every HTTP request at INFO; keep the output readable.
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    try:
        if args.command == "extract":
            return run_extract(args, settings)
    except (FileNotFoundError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
