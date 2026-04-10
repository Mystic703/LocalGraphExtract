"""
retrieval.py — Hybrid retriever, query engine, and interactive Q&A loop.
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
    p     = get_profile()
    llm   = get_answer_llm()
    embed = get_embed_model()
    top_k = p["similarity_top_k"]

    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    graph_retriever = VectorContextRetriever(
        index.property_graph_store,
        embed_model=embed,
        similarity_top_k=top_k * 2,
        path_depth=1,
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


# ── Single question ────────────────────────────────────────────────────────────

def _ask(query_engine: RetrieverQueryEngine, question: str) -> None:
    """Run one question and print the answer with source previews."""
    print(f"\n{'=' * 60}")
    print(f"Q : {question}")
    print("=" * 60)

    try:
        response = query_engine.query(question)
        print(f"\nR : {response.response}")

        if response.source_nodes:
            print(f"\nSources ({len(response.source_nodes)}) :")
            for node in response.source_nodes[:3]:
                score   = f"{node.score:.4f}" if node.score is not None else "N/A"
                preview = node.node.text[:200].replace("\n", " ")
                print(f"  [{score}] {preview}...")

    except Exception as e:
        logger.error(f"Query failed: {e}")
        print(f"  Erreur : {e}")


# ── Batch mode ─────────────────────────────────────────────────────────────────

def run_queries(query_engine: RetrieverQueryEngine, questions: list[str]) -> None:
    """Fire a predefined list of questions and print all answers."""
    for question in questions:
        _ask(query_engine, question)


# ── Interactive mode ───────────────────────────────────────────────────────────

def run_interactive(query_engine: RetrieverQueryEngine) -> None:
    """
    Start an interactive Q&A loop in the terminal.
    Type 'exit' or press Ctrl+C to quit.
    """
    print("\n" + "=" * 60)
    print("  Mode interactif — posez vos questions sur le document")
    print("  Tapez 'exit' pour quitter")
    print("=" * 60)

    while True:
        try:
            question = input("\nQ : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir.")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Au revoir.")
            break

        _ask(query_engine, question)
