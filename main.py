"""
main.py — Pipeline entry point.

Local:
    python main.py --source data/rapport.pdf          # ingest + query
    python main.py --source data/rapport.pdf --ingest # ingest only
    python main.py --query-only                        # load cached graph + query
    python main.py --source data/rapport.pdf --viz     # ingest + visualise

CCRT (called by SLURM scripts):
    PIPELINE_MODE=ccrt python main.py --source $WORK/data/rapport.pdf --ingest
    PIPELINE_MODE=ccrt python main.py --query-only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

# ── Directory bootstrap ────────────────────────────────────────────────────────
# Must happen before config is imported so env vars are already set.
for _d in ("logs", "cache"):
    Path(_d).mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
_slurm_id  = os.getenv("SLURM_JOB_ID", "local")
_log_file  = f"logs/pipeline_{_slurm_id}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Local imports ──────────────────────────────────────────────────────────────
from config import MODE, get_profile
from graph import build_graph_store, build_index, load_existing_index, print_graph_stats
from loader import load_document, chunk_markdown
from retrieval import build_query_engine, run_queries
from visualize import plot_graph

# ── Demo questions (edit freely) ──────────────────────────────────────────────
DEMO_QUESTIONS = [
    "Comment est assurée la protection contre les incendies ?",
    "Quelles sont les valeurs de relâchement admissibles en CNT et en CAT ?",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GraphRAG pipeline — document → graph → RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source",       type=str,  help="Path to the source PDF")
    p.add_argument("--ingest",       action="store_true", help="Run ingestion only (no QA)")
    p.add_argument("--query-only",   action="store_true", help="Skip ingestion, load cached graph")
    p.add_argument("--viz",          action="store_true", help="Plot the knowledge graph after ingestion")
    p.add_argument("--force-reload", action="store_true", help="Ignore doc cache, re-convert PDF")
    p.add_argument("--question",     type=str,  help="Single question (non-interactive)")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    profile = get_profile()

    logger.info(f"Pipeline start | MODE={MODE} | SLURM_JOB_ID={_slurm_id}")

    graph_store = build_graph_store()

    # ── Ingestion ──────────────────────────────────────────────────────────────
    if not args.query_only:
        if not args.source:
            logger.error("--source is required unless --query-only is set.")
            sys.exit(1)

        source_path = str(Path(args.source).resolve())
        markdown    = load_document(source_path, force_reload=args.force_reload)
        nodes       = chunk_markdown(markdown, node_slice=profile["node_slice"])

        logger.info(f"Ingesting {len(nodes)} node(s) into graph store…")
        index = build_index(nodes, graph_store)
        print_graph_stats(graph_store)

    else:
        index = load_existing_index(graph_store)
        print_graph_stats(graph_store)

    # ── Visualisation (local only — requires display) ──────────────────────────
    if args.viz:
        if MODE == "ccrt":
            logger.warning("--viz skipped: no display available on CCRT.")
        else:
            plot_graph(index)

    # ── QA ────────────────────────────────────────────────────────────────────
    if not args.ingest:
        query_engine = build_query_engine(index)
        questions    = [args.question] if args.question else DEMO_QUESTIONS
        run_queries(query_engine, questions)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
