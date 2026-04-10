"""
models.py — LLM and embedding factories.

Backend selection is driven by config.py → get_profile()["backend"]:
  "llama_cpp"  → llama-cpp-python  (CPU / consumer GPU, GGUF quantised models)
  "vllm"       → vLLM              (A100 / H100, HuggingFace models, CUDA only)
  "hf"         → HuggingFace transformers pipeline (fallback GPU, no vLLM server)

All models are loaded from local paths / offline cache — no internet call is made
at runtime (TRANSFORMERS_OFFLINE and HF_DATASETS_OFFLINE are set automatically).
"""

from __future__ import annotations

import logging
import os

from llama_index.core.llms import LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import get_profile, MODE

logger = logging.getLogger(__name__)

# ── Force offline mode on CCRT ─────────────────────────────────────────────────
if MODE == "ccrt":
    os.environ.setdefault("TRANSFORMERS_OFFLINE",   "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE",    "1")
    os.environ.setdefault("HF_HUB_OFFLINE",         "1")
    logger.info("HuggingFace offline mode enabled.")

# ── Singletons ─────────────────────────────────────────────────────────────────
_llm_extract: LLM | None = None
_llm_answer:  LLM | None = None
_embed_model: HuggingFaceEmbedding | None = None


# ── Builder helpers ────────────────────────────────────────────────────────────

def _build_llama_cpp(model_path: str, max_tokens: int) -> LLM:
    from llama_index.llms.llama_cpp import LlamaCPP
    p = get_profile()
    logger.info(f"[llama.cpp] Loading: {model_path}")
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


def _build_vllm(model_name: str, max_tokens: int, tensor_parallel: int) -> LLM:
    from llama_index.llms.vllm import Vllm
    p = get_profile()
    logger.info(f"[vLLM] Loading: {model_name}  (tp={tensor_parallel})")
    return Vllm(
        model=model_name,
        tensor_parallel_size=tensor_parallel,
        max_new_tokens=max_tokens,
        context_window=p["n_ctx"],
        dtype=p["dtype"],
        gpu_memory_utilization=p["gpu_memory_utilization"],
        temperature=0.0,
        vllm_kwargs={
            "trust_remote_code": True,
            "enforce_eager":     False,   # use CUDA graph captures for speed
        },
    )


def _build_hf(model_name: str, max_tokens: int) -> LLM:
    """Fallback: plain HuggingFace pipeline (single GPU, no tensor parallelism)."""
    from llama_index.llms.huggingface import HuggingFaceLLM
    import torch
    logger.info(f"[HF pipeline] Loading: {model_name}")
    return HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        max_new_tokens=max_tokens,
        device_map="auto",
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        },
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def get_extraction_llm() -> LLM:
    """Return (and cache) the LLM used for KG triplet extraction."""
    global _llm_extract
    if _llm_extract is None:
        p = get_profile()
        backend = p["backend"]
        if backend == "llama_cpp":
            _llm_extract = _build_llama_cpp(p["extraction_model"], p["extract_tokens"])
        elif backend == "vllm":
            _llm_extract = _build_vllm(
                p["extraction_model"], p["extract_tokens"], p["tensor_parallel"]
            )
        elif backend == "hf":
            _llm_extract = _build_hf(p["extraction_model"], p["extract_tokens"])
        else:
            raise ValueError(f"Unknown backend: {backend}")
    return _llm_extract


def get_answer_llm() -> LLM:
    """Return (and cache) the LLM used for RAG answer synthesis."""
    global _llm_answer
    if _llm_answer is None:
        p = get_profile()
        backend = p["backend"]
        if backend == "llama_cpp":
            _llm_answer = _build_llama_cpp(p["answer_model"], p["answer_tokens"])
        elif backend == "vllm":
            tp = p.get("answer_tensor_parallel", p["tensor_parallel"])
            _llm_answer = _build_vllm(p["answer_model"], p["answer_tokens"], tp)
        elif backend == "hf":
            _llm_answer = _build_hf(p["answer_model"], p["answer_tokens"])
        else:
            raise ValueError(f"Unknown backend: {backend}")
    return _llm_answer


def get_embed_model() -> HuggingFaceEmbedding:
    """Return (and cache) the shared embedding model."""
    global _embed_model
    if _embed_model is None:
        p = get_profile()
        logger.info(f"Loading embedding model: {p['embedding_model']}")
        _embed_model = HuggingFaceEmbedding(
            model_name=p["embedding_model"],
            cache_folder=p["embedding_dir"],
        )
    return _embed_model
