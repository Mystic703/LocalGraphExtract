"""
models.py — LLM and embedding model factories.
All model config is read from config.py — swap MODE there, nothing else changes.
"""

from __future__ import annotations

import logging

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import get_profile, EMBEDDING_MODEL, EMBEDDING_CACHE_DIR

logger = logging.getLogger(__name__)

# Module-level singletons (lazy-initialised)
_llm_extract: LlamaCPP | None = None
_llm_answer:  LlamaCPP | None = None
_embed_model: HuggingFaceEmbedding | None = None


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_llm(model_path: str, max_tokens: int) -> LlamaCPP:
    p = get_profile()
    logger.info(f"Loading LLM: {model_path}")
    return LlamaCPP(
        model_path=model_path,
        temperature=0.0,
        max_new_tokens=max_tokens,
        context_window=p["n_ctx"],
        model_kwargs={
            "n_threads": p["n_threads"],
            "n_batch":   p["n_batch"],
            "n_ctx":     p["n_ctx"],
        },
        verbose=False,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def get_extraction_llm() -> LlamaCPP:
    """Return (and cache) the LLM used for KG triplet extraction."""
    global _llm_extract
    if _llm_extract is None:
        p = get_profile()
        _llm_extract = _build_llm(p["extraction_model"], p["extract_tokens"])
    return _llm_extract


def get_answer_llm() -> LlamaCPP:
    """Return (and cache) the LLM used for RAG answer synthesis."""
    global _llm_answer
    if _llm_answer is None:
        p = get_profile()
        _llm_answer = _build_llm(p["answer_model"], p["answer_tokens"])
    return _llm_answer


def get_embed_model() -> HuggingFaceEmbedding:
    """Return (and cache) the shared embedding model."""
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            cache_folder=EMBEDDING_CACHE_DIR,
        )
    return _embed_model
