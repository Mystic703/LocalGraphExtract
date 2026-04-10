"""
config.py — Central configuration.
Set MODE before submitting a SLURM job. Everything else is automatic.

Modes:
  "local_test"  — CPU laptop, llama.cpp GGUF, Neo4j via Docker
  "ccrt"        — CEA CCRT A100s, vLLM + HuggingFace, SimplePropertyGraphStore (no server)
"""

import os

# ── Active mode ────────────────────────────────────────────────────────────────
MODE = os.getenv("PIPELINE_MODE", "local_test")   # override via env var or edit directly

# ── Shared paths ───────────────────────────────────────────────────────────────
DATA_DIR       = os.getenv("PIPELINE_DATA_DIR",   "data")
MODELS_DIR     = os.getenv("PIPELINE_MODELS_DIR", "models")
CACHE_DIR      = os.getenv("PIPELINE_CACHE_DIR",  "cache")
LOG_DIR        = os.getenv("PIPELINE_LOG_DIR",    "logs")

DOC_CACHE_FILE   = f"{CACHE_DIR}/doc_cache.md"
TRIPLET_LOG      = f"{LOG_DIR}/triplet_log.jsonl"
GRAPH_STORE_PATH = f"{CACHE_DIR}/graph_store.json"   # used by SimplePropertyGraphStore

# ── Neo4j (local_test only) ───────────────────────────────────────────────────
NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "nonenone"

# ── Per-mode profiles ──────────────────────────────────────────────────────────
_PROFILES = {

    # ── Laptop / workstation ───────────────────────────────────────────────────
    "local_test": {
        "backend":          "llama_cpp",             # "llama_cpp" | "vllm" | "hf"
        "graph_store":      "neo4j",                 # "neo4j" | "simple"
        "extraction_model": f"{MODELS_DIR}/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "answer_model":     f"{MODELS_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "embedding_model":  "BAAI/bge-small-en-v1.5",
        "embedding_dir":    f"{MODELS_DIR}/embeddings",
        # llama.cpp specific
        "n_threads":        4,
        "n_batch":          512,
        "n_ctx":            4096,
        "extract_tokens":   512,
        "answer_tokens":    1024,
        # pipeline
        "max_paths":        5,
        "num_workers":      1,
        "similarity_top_k": 3,
        "node_slice":       (15, 20),
    },

    # ── CEA CCRT — A100 GPUs ───────────────────────────────────────────────────
    # All paths point to $WORK (persistent) or $SCRATCH (fast, purged after 30d).
    # Models must be pre-downloaded with scripts/download_models.py before going offline.
    "ccrt": {
        "backend":          "vllm",
        "graph_store":      "simple",                # file-based, no server required
        "extraction_model": "meta-llama/Llama-3.2-3B-Instruct",
        "answer_model":     "meta-llama/Llama-3.1-70B-Instruct",
        "embedding_model":  "BAAI/bge-small-en-v1.5",
        "embedding_dir":    f"{MODELS_DIR}/embeddings",
        # vLLM specific
        "tensor_parallel":  1,   # GPUs for extraction model  (3B fits on 1 × A100-80G)
        "answer_tensor_parallel": 4,  # GPUs for answer model (70B needs 4 × A100-80G with Q4)
        "gpu_memory_utilization": 0.90,
        "n_ctx":            8192,
        "extract_tokens":   1024,
        "answer_tokens":    2048,
        "dtype":            "bfloat16",              # A100 native format
        # pipeline
        "max_paths":        15,
        "num_workers":      4,
        "similarity_top_k": 10,
        "node_slice":       None,                    # process all nodes
    },
}


def get_profile() -> dict:
    profile = _PROFILES.get(MODE)
    if profile is None:
        raise ValueError(f"Unknown MODE '{MODE}'. Available: {list(_PROFILES)}")
    return profile
