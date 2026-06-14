#!/bin/bash
#SBATCH --job-name=emb_e5lg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%j-generate_e5_title_embeddings.out
#SBATCH --error=logs/%j-generate_e5_title_embeddings.err

set -euo pipefail

export HF_HOME=/home/solbeecho/data/datasets/huggingface_cache
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

source /home/solbeecho/data/.bashrc
source /home/solbeecho/data/miniconda3/etc/profile.d/conda.sh
conda activate wikirun
export LD_LIBRARY_PATH="/data6/solbeecho/.conda/envs/wikirun/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO_DIR"
mkdir -p outputs logs

echo "[$(date '+%H:%M:%S')] START generate E5-large title passage embeddings"
srun python generate_embeddings.py --embedding-config config/embed/e5-large-title-passage.yaml

echo "[$(date '+%H:%M:%S')] START generate E5-large title query embeddings"
srun python generate_embeddings.py --embedding-config config/embed/e5-large-title-query.yaml

echo "[$(date '+%H:%M:%S')] DONE generate E5-large title embeddings"
