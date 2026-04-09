"""
graph.py — Neo4j store setup, index building, and triplet validation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List

from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.schema import BaseNode
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, TRIPLET_LOG, MODE, get_profile
from models import get_extraction_llm, get_answer_llm, get_embed_model
from prompts import EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

# Matches numeric values that might be hallucinated
_NUMERIC_RE = re.compile(
    r"\b\d[\d.,]*\s*(?:°C|K|mm|cm|m²|m|kg|g|bar|MPa|kPa|%|A|W|h|s|min|ans|ans)?\b"
)


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
        llm=get_extraction_llm(),
        extract_prompt=EXTRACTION_PROMPT,
        max_paths_per_chunk=p["max_paths"],
        num_workers=p["num_workers"],
    )


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_node_triplets(graph_store: Neo4jPropertyGraphStore, source_text: str) -> None:
    """
    After inserting a node, remove any newly added relation whose numeric
    object value cannot be found verbatim in the source text.
    """
    graph = graph_store.graph
    to_remove = []

    for rel_id, rel in graph.relations.items():
        obj_text = str(rel.target_id)
        if _NUMERIC_RE.search(obj_text) and obj_text not in source_text:
            logger.warning(f"  ⚠ Hallucinated triplet dropped: "
                           f"{rel.source_id} —[{rel.label}]→ {obj_text}")
            to_remove.append(rel_id)

    for rel_id in to_remove:
        graph.relations.pop(rel_id, None)


# ── Logging ───────────────────────────────────────────────────────────────────

def _log_triplets(node_index: int, source_text: str, graph_store: Neo4jPropertyGraphStore) -> None:
    Path(TRIPLET_LOG).parent.mkdir(parents=True, exist_ok=True)
    graph = graph_store.graph
    triplets = [
        {"source": r.source_id, "relation": r.label, "target": r.target_id}
        for r in graph.relations.values()
    ]
    entry = {
        "timestamp":      datetime.now().isoformat(),
        "node_index":     node_index,
        "source_preview": source_text[:200],
        "triplet_count":  len(triplets),
        "triplets":       triplets,
        "mode":           MODE,
    }
    with open(TRIPLET_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Index building ─────────────────────────────────────────────────────────────

def build_index(nodes: List[BaseNode], graph_store: Neo4jPropertyGraphStore) -> PropertyGraphIndex:
    """
    Build (or incrementally update) a PropertyGraphIndex from a list of nodes.
    Validates each node's triplets against the source text before committing.

    Returns the final PropertyGraphIndex.
    """
    extractor = build_extractor()
    embed    = get_embed_model()
    llm      = get_answer_llm()
    index    = None

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

            _validate_node_triplets(graph_store, node.text)
            _log_triplets(i, node.text, graph_store)

            logger.info(f"  ✓ Node {i} done")
            time.sleep(1)

        except Exception as exc:
            logger.error(f"  ✗ Node {i} skipped: {exc}")
            continue

    if index is None:
        raise RuntimeError("No nodes were successfully processed.")

    return index


def load_existing_index(graph_store: Neo4jPropertyGraphStore) -> PropertyGraphIndex:
    """Load an already-populated index from Neo4j (skip re-extraction)."""
    logger.info("Loading existing index from Neo4j...")
    return PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=get_embed_model(),
        llm=get_answer_llm(),
        show_progress=True,
    )


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_graph_stats(graph_store: Neo4jPropertyGraphStore) -> None:
    client = graph_store.client
    records, _, _ = client.execute_query("MATCH (n) RETURN count(n) AS count")
    n_nodes = records[0]["count"]

    vec_res, _, _ = client.execute_query(
        "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) AS c"
    )
    n_vecs = vec_res[0]["c"]

    logger.info(f"Graph stats — nodes: {n_nodes} | with embeddings: {n_vecs}")
    print(f"  Nodes total     : {n_nodes}")
    print(f"  Nodes w/ vectors: {n_vecs}")
