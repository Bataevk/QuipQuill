"""Command line entry point: `uv run builder <command>`."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from quipquill_builder import curation, storyfile
from quipquill_builder.config import Settings
from quipquill_builder.embeddings import OpenRouterBatchEmbedder, OpenRouterEmbedder
from quipquill_builder.export import ExportError, export_story
from quipquill_builder.extractor import ExtractionError, ExtractionParams, GraphExtractor
from quipquill_builder.llm import OpenRouterBatchClient, OpenRouterChatClient
from quipquill_builder.meta_gen import draft_meta
from quipquill_builder.models import Graph
from quipquill_builder.text import load_documents
from quipquill_builder.validate import Issue, has_errors, validate


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
    extract.add_argument(
        "--sync",
        action="store_true",
        default=not settings.use_batch,
        help="regular calls via the allowed providers instead of the Batch API (faster, 2x price)",
    )
    extract.add_argument(
        "--resume-batch",
        metavar="ID",
        help="wait for an already submitted extraction batch instead of submitting a new one",
    )

    curate = commands.add_parser("curate", help="apply manual fixes (TOML) to graph.json")
    curate.add_argument("graph", type=Path, help="graph.json from `extract`")
    curate.add_argument("rules", type=Path, help="curation.toml")
    curate.add_argument("-o", "--output", type=Path, required=True)

    export = commands.add_parser("export", help="graph.json → .qqstory (with embeddings)")
    export.add_argument("graph", type=Path, help="graph.json from `extract`")
    export.add_argument("-o", "--output", type=Path, required=True, help="path to .qqstory")
    export.add_argument("--title", help="story title (default: output file name)")
    export.add_argument("--start", help="start location: name or id (default: first location)")
    export.add_argument("--language", default="en")
    export.add_argument("--embed-model", default=settings.embed_model)
    export.add_argument(
        "--sync",
        action="store_true",
        default=not settings.use_batch,
        help="direct embeddings calls instead of the Batch API",
    )

    check = commands.add_parser("validate", help="check that a .qqstory is playable")
    check.add_argument("story", type=Path)
    check.add_argument(
        "--draft", action="store_true", help="treat empty world_summary/constraints/style as OK"
    )

    meta = commands.add_parser("meta", help="draft or edit world_summary, world_constraints, style")
    meta.add_argument("story", type=Path)
    meta.add_argument("--show", action="store_true", help="print meta and exit")
    meta.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set a meta value by hand (repeatable)",
    )
    meta.add_argument("--force", action="store_true", help="redraft keys that are already set")
    meta.add_argument("--model", default=settings.extract_model)
    return parser


def print_issues(issues: list[Issue]) -> None:
    for issue in issues:
        print(issue)
    errors = sum(issue.level == "error" for issue in issues)
    print(f"{errors} errors, {len(issues) - errors} warnings")


def run_extract(args: argparse.Namespace, settings: Settings) -> int:
    documents = load_documents(args.input_dir)
    if not documents:
        logging.error("No .md/.txt files in %s", args.input_dir)
        return 1
    if args.sync:
        if args.resume_batch:
            logging.error("--resume-batch cannot be combined with --sync")
            return 1
        extractor_client = OpenRouterChatClient(settings, args.model)
        make_summarizer = OpenRouterChatClient
    else:
        extractor_client = OpenRouterBatchClient(settings, args.model, resume_id=args.resume_batch)
        make_summarizer = OpenRouterBatchClient
    summarizer_client = (
        make_summarizer(settings, args.summarize_model)
        if args.summarize_model and args.summarize_model != args.model
        else extractor_client
    )
    logging.info("Mode: %s", "sync" if args.sync else "batch (may take minutes)")
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


def run_curate(args: argparse.Namespace) -> int:
    graph = Graph.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
    curated = curation.apply(graph, curation.load(args.rules))
    args.output.write_text(
        json.dumps(curated.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{len(graph.entities)} → {len(curated.entities)} entities, "
        f"{len(graph.relationships)} → {len(curated.relationships)} relationships → {args.output}"
    )
    return 0


def run_export(args: argparse.Namespace, settings: Settings) -> int:
    graph = Graph.from_dict(json.loads(args.graph.read_text(encoding="utf-8")))
    embedder_type = OpenRouterEmbedder if args.sync else OpenRouterBatchEmbedder
    embedder = embedder_type(settings, args.embed_model)
    title = args.title or args.output.stem.replace("-", " ").replace("_", " ").title()
    try:
        issues = asyncio.run(
            export_story(
                graph,
                args.output,
                embedder,
                title=title,
                start=args.start,
                language=args.language,
            )
        )
    except ExportError as error:
        logging.error("%s", error)
        return 1
    print_issues(issues)
    if has_errors(issues):
        print("Export aborted: fix the graph and run again. Nothing was written.")
        return 1
    print(f"Wrote {args.output}. Next: `builder meta {args.output}` to draft the world texts.")
    return 0


def run_validate(args: argparse.Namespace) -> int:
    if not args.story.is_file():
        raise FileNotFoundError(f"Story file not found: {args.story}")
    with storyfile.connect(args.story) as conn:
        issues = validate(conn, require_authored_meta=not args.draft)
    print_issues(issues)
    return 1 if has_errors(issues) else 0


def run_meta(args: argparse.Namespace, settings: Settings) -> int:
    if not args.story.is_file():
        raise FileNotFoundError(f"Story file not found: {args.story}")
    with storyfile.connect(args.story) as conn:
        current = storyfile.read_meta(conn)
        if args.show:
            for key in sorted(current):
                print(f"{key}: {current[key]}")
            return 0

        if args.set:
            updates = {}
            for item in args.set:
                key, sep, value = item.partition("=")
                if not sep or not key.strip():
                    logging.error("--set expects KEY=VALUE, got %r", item)
                    return 1
                updates[key.strip()] = value
            storyfile.set_meta(conn, updates)
            print(f"Updated: {', '.join(updates)}")
            return 0

        todo = [k for k in storyfile.AUTHORED_META if args.force or not current.get(k, "").strip()]
        if not todo:
            print("World texts are already set; use --force to redraft or --set to edit.")
            return 0
        draft = asyncio.run(
            draft_meta(OpenRouterChatClient(settings, args.model), conn, current.get("title", ""))
        )
        storyfile.set_meta(conn, {key: draft[key] for key in todo})
    for key in todo:
        print(f"--- {key}\n{draft[key]}\n")
    print("Draft saved. Edit with: builder meta <story> --set style='...'")
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
        match args.command:
            case "extract":
                return run_extract(args, settings)
            case "curate":
                return run_curate(args)
            case "export":
                return run_export(args, settings)
            case "validate":
                return run_validate(args)
            case "meta":
                return run_meta(args, settings)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logging.error("%s", error)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
