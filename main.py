"""
main.py — Orchestrates the full pipeline end-to-end.

Usage:
    python main.py --source path/to/doc.pdf          # full run (ingest + query)
    python main.py --source path/to/doc.pdf --ingest # ingest only
    python main.py --query-only                      # load existing index + query
    python main.py --source path/to/doc.pdf --viz    # ingest + visualise graph
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

# ── Local imports (after logging is configured) ────────────────────────────────
from config import MODE, get_profile
from graph import build_graph_store, build_index, load_existing_index, print_graph_stats
from loader import chunk_markdown, load_document
from retrieval import build_query_engine, run_queries
from visualize import plot_graph

# ── Sample questions (edit freely) ────────────────────────────────────────────
DEMO_QUESTIONS = [
    "Comment est assurée la protection contre les incendies ?",
    "Quelles sont les valeurs de relâchement admissibles en CNT et en CAT ?",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KG pipeline — document → Neo4j → RAG")
    p.add_argument("--source",     type=str,  help="Path to the source PDF")
    p.add_argument("--ingest",     action="store_true", help="Run ingestion only, skip QA")
    p.add_argument("--query-only", action="store_true", help="Skip ingestion, load existing index")
    p.add_argument("--viz",        action="store_true", help="Visualise the graph after ingestion")
    p.add_argument("--force-reload", action="store_true", help="Ignore doc cache and re-convert PDF")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_profile()

    logger.info(f"Starting pipeline  |  MODE={MODE}")

    graph_store = build_graph_store()

    # ── Ingestion ──────────────────────────────────────────────────────────────
    if not args.query_only:
        if not args.source:
            logger.error("--source is required unless --query-only is set.")
            sys.exit(1)

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

    # ── QA ────────────────────────────────────────────────────────────────────
    if not args.ingest:
        query_engine = build_query_engine(index)
        run_queries(query_engine, DEMO_QUESTIONS)


if __name__ == "__main__":
    main()
