#!/bin/bash
#SBATCH --job-name=graphrag-ingest
#SBATCH --partition=gpu_a100            # adjust to CCRT partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                    # 4 × A100-80G (needed for 70B answer model)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ingest_%j.out
#SBATCH --error=logs/ingest_%j.err
#SBATCH --account=<YOUR_CCRT_ACCOUNT>   # ← fill in

# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load python/3.11
module load cuda/12.2
module load gcc/11.3

# Project root (edit to match your $WORK path)
PROJECT_DIR="$WORK/GraphRag_Local"
cd "$PROJECT_DIR" || exit 1

source .venv/bin/activate

# ── Offline mode — no outbound network during jobs on CCRT ────────────────────
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Pipeline config ────────────────────────────────────────────────────────────
export PIPELINE_MODE=ccrt
export PIPELINE_DATA_DIR="$WORK/data"
export PIPELINE_MODELS_DIR="$WORK/models"
export PIPELINE_CACHE_DIR="$SCRATCH/graphrag_cache"
export PIPELINE_LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$PIPELINE_CACHE_DIR" "$PIPELINE_LOG_DIR"

# ── Source PDF (pass as argument or hardcode) ──────────────────────────────────
SOURCE_PDF="${1:-$WORK/data/rapport.pdf}"

echo "=========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Source PDF  : $SOURCE_PDF"
echo "Mode        : $PIPELINE_MODE"
echo "=========================================="

python main.py --source "$SOURCE_PDF" --ingest

echo "Ingestion finished — exit code $?"
