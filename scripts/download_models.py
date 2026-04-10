"""
download_models.py — Pre-download all HuggingFace models to local storage.

Run this ONCE on a machine with internet access (e.g. a CCRT login node
before your job starts, or your local workstation before transferring files).

Usage:
    # On a machine with internet:
    python scripts/download_models.py --mode ccrt --models-dir $WORK/models

    # Then transfer to CCRT if needed:
    rsync -av models/ ccrt:$WORK/models/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def download_hf_model(model_id: str, cache_dir: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"  → {model_id}")
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        ignore_patterns=["*.bin", "*.pt"],   # prefer safetensors
    )
    print(f"    ✓ saved to {cache_dir}")


def download_embedding(model_id: str, cache_dir: str) -> None:
    from sentence_transformers import SentenceTransformer
    print(f"  → {model_id}  (embedding)")
    SentenceTransformer(model_id, cache_folder=cache_dir)
    print(f"    ✓ saved to {cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download models for offline CCRT use")
    parser.add_argument("--mode",       default="ccrt",    help="Profile from config.py")
    parser.add_argument("--models-dir", default="models",  help="Local directory to store models")
    parser.add_argument("--hf-token",   default=None,      help="HuggingFace token (for gated models)")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token
        print(f"HuggingFace token set.")

    # Import config profile
    sys.path.insert(0, str(Path(__file__).parent.parent))
    os.environ["PIPELINE_MODE"] = args.mode
    from config import get_profile
    p = get_profile()

    hf_cache = str(models_dir)

    print(f"\nDownloading models for MODE='{args.mode}' → {models_dir}\n")

    models_to_download = []

    if p["backend"] in ("vllm", "hf"):
        models_to_download += [
            p["extraction_model"],
            p["answer_model"],
        ]
    elif p["backend"] == "llama_cpp":
        print("llama_cpp backend uses local GGUF files — download them manually from HuggingFace.")
        print("Example:")
        print("  huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF "
              "Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/")
        sys.exit(0)

    print("LLM models:")
    for m in models_to_download:
        download_hf_model(m, hf_cache)

    print("\nEmbedding model:")
    download_embedding(p["embedding_model"], str(models_dir / "embeddings"))

    print(f"\n✓ All models saved to {models_dir}")
    print("  You can now set TRANSFORMERS_OFFLINE=1 / HF_HUB_OFFLINE=1 safely.")


if __name__ == "__main__":
    main()
