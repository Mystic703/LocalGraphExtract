"""
retrieval.py — Hybrid retriever and query engine construction.
"""

from __future__ import annotations

import logging

from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import VectorContextRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever

from config import get_profile
from models import get_answer_llm, get_embed_model
from prompts import EXPERT_PROMPT

logger = logging.getLogger(__name__)


def build_query_engine(index: PropertyGraphIndex) -> RetrieverQueryEngine:
    """
    Build a hybrid (vector + graph) query engine.

    Strategy:
      - VectorContextRetriever  → finds nodes by semantic similarity + walks
                                  graph edges (path_depth=1)
      - index.as_retriever      → standard vector retriever over node text
      - QueryFusionRetriever    → deduplicates and re-ranks results from both
    """
    p = get_profile()
    llm   = get_answer_llm()
    embed = get_embed_model()
    top_k = p["similarity_top_k"]

    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    graph_retriever = VectorContextRetriever(
        index.property_graph_store,
        embed_model=embed,
        similarity_top_k=top_k * 2,
        path_depth=1,        # keep focused; raise to 2 on server if needed
        include_text=True,
    )

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, graph_retriever],
        similarity_top_k=top_k + 3,
        llm=llm,
        num_queries=1,
        use_async=False,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llm,
        text_qa_template=EXPERT_PROMPT,
    )

    logger.info("Query engine ready.")
    return query_engine


def run_queries(query_engine: RetrieverQueryEngine, questions: list[str]) -> None:
    """Run a list of questions and pretty-print answers + source previews."""
    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"Q : {question}")
        print("=" * 60)

        response = query_engine.query(question)
        print(f"R : {response.response}")

        print(f"\nSources utilisées : {len(response.source_nodes)}")
        for node in response.source_nodes[:3]:
            score = f"{node.score:.4f}" if node.score is not None else "N/A"
            print(f"  [{score}] {node.node.text[:200]}")
