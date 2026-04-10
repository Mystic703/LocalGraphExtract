"""
graph.py — Neo4j store setup and index building.
Matches notebook behaviour: extraction LLM is used for both kg_extractors and index llm.
"""

from __future__ import annotations

import logging
import time
from typing import List

from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.schema import BaseNode
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, get_profile
from models import get_extraction_llm, get_answer_llm, get_embed_model
from prompts import EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


# ── Store ──────────────────────────────────────────────────────────────────────

def build_graph_store() -> Neo4jPropertyGraphStore:
    logger.info(f"Connecting to Neo4j at {NEO4J_URI}")
    return Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
    )


# ── Extractor ──────────────────────────────────────────────────────────────────

def build_extractor() -> SimpleLLMPathExtractor:
    p = get_profile()
    return SimpleLLMPathExtractor(
        llm=get_extraction_llm(),   # 3B for extraction, matching notebook
        extract_prompt=EXTRACTION_PROMPT,
        max_paths_per_chunk=p["max_paths"],
        num_workers=p["num_workers"],
    )


# ── Index building ─────────────────────────────────────────────────────────────

def build_index(nodes: List[BaseNode], graph_store: Neo4jPropertyGraphStore) -> PropertyGraphIndex:
    """
    Matches notebook exactly:
    - kg_extractors uses extraction LLM (3B)
    - PropertyGraphIndex llm also uses extraction LLM (3B), not answer LLM
    - insert_nodes for subsequent nodes
    """
    extractor = build_extractor()
    embed     = get_embed_model()
    llm       = get_extraction_llm()  # ← matches notebook: llm_extract for index too
    index     = None

    for i, node in enumerate(nodes):
        logger.info(f"[{i + 1}/{len(nodes)}] Processing node...")
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

            logger.info(f"  ✓ Node {i} done")
            time.sleep(1)

        except Exception as exc:
            logger.error(f"  ✗ Node {i} skipped: {exc}")
            continue

    if index is None:
        raise RuntimeError("No nodes were successfully processed.")

    return index


# ── Load existing ──────────────────────────────────────────────────────────────

def load_existing_index(graph_store: Neo4jPropertyGraphStore) -> PropertyGraphIndex:
    """Load an already-populated index from Neo4j (skip re-extraction)."""
    logger.info("Loading existing index from Neo4j...")
    return PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=get_embed_model(),
        llm=get_answer_llm(),   # answer LLM for QA, not extraction
        show_progress=True,
    )


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_graph_stats(graph_store: Neo4jPropertyGraphStore) -> None:
    try:
        records, _, _ = graph_store.client.execute_query(
            "MATCH (n) RETURN count(n) AS count"
        )
        n_nodes = records[0]["count"]

        vec_res, _, _ = graph_store.client.execute_query(
            "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
        )
        n_vecs = vec_res[0]["c"]

        logger.info(f"Graph stats — nodes: {n_nodes} | with embeddings: {n_vecs}")
        print(f"  Nodes total     : {n_nodes}")
        print(f"  Nodes w/ vectors: {n_vecs}")
    except Exception as e:
        logger.warning(f"Could not fetch graph stats: {e}")