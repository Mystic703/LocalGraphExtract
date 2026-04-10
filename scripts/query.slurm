#!/bin/bash
#SBATCH --job-name=graphrag-query
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                    # 4 × A100-80G for 70B answer model
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/query_%j.out
#SBATCH --error=logs/query_%j.err
#SBATCH --account=<YOUR_CCRT_ACCOUNT>   # ← fill in

# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load python/3.11
module load cuda/12.2
module load gcc/11.3

PROJECT_DIR="$WORK/GraphRag_Local"
cd "$PROJECT_DIR" || exit 1

source .venv/bin/activate

# ── Offline mode ───────────────────────────────────────────────────────────────
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Pipeline config ────────────────────────────────────────────────────────────
export PIPELINE_MODE=ccrt
export PIPELINE_DATA_DIR="$WORK/data"
export PIPELINE_MODELS_DIR="$WORK/models"
export PIPELINE_CACHE_DIR="$SCRATCH/graphrag_cache"
export PIPELINE_LOG_DIR="$PROJECT_DIR/logs"

# ── Question (pass as SLURM argument or set here) ─────────────────────────────
# Usage: sbatch scripts/query.slurm "your question here"
QUESTION="${1:-Comment est assurée la protection contre les incendies ?}"

echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Question : $QUESTION"
echo "=========================================="

python main.py --query-only --question "$QUESTION"

echo "Query finished — exit code $?"
