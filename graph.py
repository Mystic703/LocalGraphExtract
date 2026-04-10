"""
graph.py — Graph store setup and index building.

Graph store backend (driven by config):
  "neo4j"   → Neo4jPropertyGraphStore   (requires running Neo4j server, local_test)
  "simple"  → SimplePropertyGraphStore  (file-based JSON, no server, used on CCRT)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

from llama_index.core import PropertyGraphIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.schema import BaseNode

from config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    GRAPH_STORE_PATH, get_profile,
)
from models import get_extraction_llm, get_answer_llm, get_embed_model
from prompts import EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

# Type alias so callers don't need to import both store types
GraphStore = SimplePropertyGraphStore  # widened at runtime to Neo4jPropertyGraphStore too


# ── Store ──────────────────────────────────────────────────────────────────────

def build_graph_store() -> GraphStore:
    p = get_profile()
    backend = p["graph_store"]

    if backend == "neo4j":
        from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
        logger.info(f"Connecting to Neo4j at {NEO4J_URI}")
        return Neo4jPropertyGraphStore(
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            url=NEO4J_URI,
        )

    elif backend == "simple":
        store_path = Path(GRAPH_STORE_PATH)
        if store_path.exists():
            logger.info(f"Loading SimplePropertyGraphStore from {store_path}")
            return SimplePropertyGraphStore.from_persist_path(str(store_path))
        else:
            logger.info("Creating new SimplePropertyGraphStore (in-memory, will persist after build)")
            return SimplePropertyGraphStore()

    else:
        raise ValueError(f"Unknown graph_store backend: '{backend}'")


def persist_graph_store(graph_store: GraphStore) -> None:
    """Flush SimplePropertyGraphStore to disk. No-op for Neo4j (it writes directly)."""
    if isinstance(graph_store, SimplePropertyGraphStore):
        Path(GRAPH_STORE_PATH).parent.mkdir(parents=True, exist_ok=True)
        graph_store.persist(persist_path=GRAPH_STORE_PATH)
        logger.info(f"Graph store persisted → {GRAPH_STORE_PATH}")


# ── Extractor ──────────────────────────────────────────────────────────────────

def build_extractor() -> SimpleLLMPathExtractor:
    p = get_profile()
    return SimpleLLMPathExtractor(
        llm=get_extraction_llm(),
        extract_prompt=EXTRACTION_PROMPT,
        max_paths_per_chunk=p["max_paths"],
        num_workers=p["num_workers"],
    )


# ── Index building ─────────────────────────────────────────────────────────────

def build_index(nodes: List[BaseNode], graph_store: GraphStore) -> PropertyGraphIndex:
    """
    Build a PropertyGraphIndex node-by-node.
    Uses extraction LLM for both kg_extractors and the index LLM (matches notebook).
    Persists SimplePropertyGraphStore to disk after each node to allow resuming.
    """
    extractor = build_extractor()
    embed     = get_embed_model()
    llm       = get_extraction_llm()
    index     = None

    for i, node in enumerate(nodes):
        logger.info(f"[{i + 1}/{len(nodes)}] Processing node…")
        try:
            if index is None:
                index = PropertyGraphIndex(
                    [node],
                    property_graph_store=graph_store,
                    kg_extractors=[extractor],
                    embed_model=embed,
                    llm=llm,
                    show_progress=True,
                    include_text=True,
                    use_async=False,
                )
            else:
                index.insert_nodes([node])

            # Checkpoint after every node (crash-safe on long HPC jobs)
            persist_graph_store(graph_store)
            logger.info(f"  ✓ Node {i} done")
            time.sleep(0.5)

        except Exception as exc:
            logger.error(f"  ✗ Node {i} skipped: {exc}")
            continue

    if index is None:
        raise RuntimeError("No nodes were successfully processed.")

    return index


def load_existing_index(graph_store: GraphStore) -> PropertyGraphIndex:
    """Load a pre-built index without re-running extraction."""
    logger.info("Loading existing index from graph store…")
    return PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=get_embed_model(),
        llm=get_answer_llm(),
        show_progress=True,
    )


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_graph_stats(graph_store: GraphStore) -> None:
    try:
        if isinstance(graph_store, SimplePropertyGraphStore):
            n_nodes = len(graph_store.graph.nodes)
            n_rels  = len(graph_store.graph.relations)
            print(f"  Nodes    : {n_nodes}")
            print(f"  Relations: {n_rels}")
        else:
            # Neo4j path
            records, _, _ = graph_store.client.execute_query(
                "MATCH (n) RETURN count(n) AS count"
            )
            n_nodes = records[0]["count"]
            vec_res, _, _ = graph_store.client.execute_query(
                "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
            )
            n_vecs = vec_res[0]["c"]
            print(f"  Nodes total     : {n_nodes}")
            print(f"  Nodes w/ vectors: {n_vecs}")
    except Exception as e:
        logger.warning(f"Could not fetch graph stats: {e}")
