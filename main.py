"""
main.py — Orchestrates the full pipeline end-to-end.

Usage:
    python main.py --source path/to/doc.pdf              # ingest + demo questions
    python main.py --source path/to/doc.pdf --interactive # ingest + interactive Q&A
    python main.py --interactive                          # load existing index + interactive Q&A
    python main.py --query-only                           # load existing index + demo questions
    python main.py --source path/to/doc.pdf --ingest      # ingest only, no Q&A
    python main.py --source path/to/doc.pdf --viz         # ingest + visualise graph
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from config import MODE, get_profile
from graph import build_graph_store, build_index, load_existing_index, print_graph_stats
from loader import chunk_markdown, load_document
from retrieval import build_query_engine, run_queries, run_interactive
from visualize import plot_graph

# ── Demo questions (used when --interactive is not set) ───────────────────────
DEMO_QUESTIONS = [
    "Comment est assurée la protection contre les incendies ?",
    "Quelles sont les valeurs de relâchement admissibles en CNT et en CAT ?",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KG pipeline — document → Neo4j → RAG")
    p.add_argument("--source",       type=str,            help="Path to the source PDF")
    p.add_argument("--ingest",       action="store_true", help="Run ingestion only, skip Q&A")
    p.add_argument("--query-only",   action="store_true", help="Skip ingestion, load existing index")
    p.add_argument("--interactive",  action="store_true", help="Start interactive Q&A loop")
    p.add_argument("--viz",          action="store_true", help="Visualise the graph after ingestion")
    p.add_argument("--force-reload", action="store_true", help="Ignore doc cache and re-convert PDF")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_profile()

    logger.info(f"Starting pipeline  |  MODE={MODE}")

    # Create output dirs if needed
    Path("logs").mkdir(exist_ok=True)
    Path("cache").mkdir(exist_ok=True)

    graph_store = build_graph_store()

    # ── Ingestion ──────────────────────────────────────────────────────────────
    if not args.query_only:
        if not args.source:
            # No source given — skip ingestion silently if --interactive
            if args.interactive:
                logger.info("No --source given, loading existing index...")
                index = load_existing_index(graph_store)
            else:
                logger.error("--source is required unless --query-only or --interactive is set.")
                sys.exit(1)
        else:
            source_path = str(Path(args.source).resolve())
            markdown    = load_document(source_path, force_reload=args.force_reload)
            nodes       = chunk_markdown(markdown, node_slice=profile["node_slice"])

            logger.info(f"Ingesting {len(nodes)} nodes into Neo4j...")
            index = build_index(nodes, graph_store)
            print_graph_stats(graph_store)
    else:
        index = load_existing_index(graph_store)
        print_graph_stats(graph_store)

    # ── Visualisation ──────────────────────────────────────────────────────────
    if args.viz:
        logger.info("Rendering graph...")
        plot_graph(index)

    # ── Q&A ───────────────────────────────────────────────────────────────────
    if not args.ingest:
        query_engine = build_query_engine(index)

        if args.interactive:
            run_interactive(query_engine)
        else:
            run_queries(query_engine, DEMO_QUESTIONS)


if __name__ == "__main__":
    main()
