"""
config.py — Central configuration. Change MODE to switch between environments.
"""

MODE = "local_test"  # "local_test" | "local_server"

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
MODELS_DIR      = "models"
CACHE_DIR       = "cache"
LOG_DIR         = "logs"

DOC_CACHE_FILE  = f"{CACHE_DIR}/doc_cache.md"
TRIPLET_LOG     = f"{LOG_DIR}/triplet_log.jsonl"

# ── Neo4j ──────────────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "nonenone"

# ── Embedding model (shared across modes) ─────────────────────────────────────
EMBEDDING_MODEL      = "BAAI/bge-small-en-v1.5"
EMBEDDING_CACHE_DIR  = f"{MODELS_DIR}/embeddings"

# ── Per-mode settings ──────────────────────────────────────────────────────────
_PROFILES = {
    "local_test": {
        "extraction_model": f"{MODELS_DIR}/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "answer_model":     f"{MODELS_DIR}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "n_threads":        4,
        "n_batch":          512,
        "n_ctx":            4096,
        "extract_tokens":   512,
        "answer_tokens":    1024,
        "max_paths":        5,
        "num_workers":      1,
        "similarity_top_k": 3,
        "node_slice":       (15, 20),
    },
    "local_server": {
        "extraction_model": f"{MODELS_DIR}/Mistral-22B-Instruct-Q5_K_M.gguf",
        "answer_model":     f"{MODELS_DIR}/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "n_threads":        16,
        "n_batch":          2048,
        "n_ctx":            8192,
        "extract_tokens":   1024,
        "answer_tokens":    2048,
        "max_paths":        15,
        "num_workers":      4,
        "similarity_top_k": 10,
        "node_slice":       None,
    },
}

def get_profile() -> dict:
    profile = _PROFILES.get(MODE)
    if profile is None:
        raise ValueError(f"Unknown MODE '{MODE}'. Choose from: {list(_PROFILES)}")
    return profile
