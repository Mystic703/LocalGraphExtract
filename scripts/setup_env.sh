#!/bin/bash
# setup_env.sh — Create the Python virtual environment on CCRT.
# Run once on a login node (internet access available).
#
# Usage:
#   bash scripts/setup_env.sh

set -euo pipefail

PROJECT_DIR="${WORK}/GraphRag_Local"
cd "$PROJECT_DIR"

module purge
module load python/3.11
module load cuda/12.2
module load gcc/11.3

echo "Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo "Installing HPC requirements..."
pip install -r requirements-hpc.txt

echo ""
echo "✓ Environment ready at $PROJECT_DIR/.venv"
echo ""
echo "Next steps:"
echo "  1.  Activate:  source .venv/bin/activate"
echo "  2.  Download models (login node, with internet):"
echo "        python scripts/download_models.py --mode ccrt --models-dir \$WORK/models"
echo "  3.  Copy PDF to \$WORK/data/"
echo "  4.  Submit ingestion job:"
echo "        sbatch scripts/ingest.slurm"
echo "  5.  After ingestion completes, submit query job:"
echo "        sbatch scripts/query.slurm"
