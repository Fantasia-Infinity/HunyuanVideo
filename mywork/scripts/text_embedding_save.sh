#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=13:00:00
#SBATCH --job-name=text_emb_11_2179
#SBATCH --output=/home/szhang2/slurm_report/slurm-%j.out
#SBATCH --error=/home/szhang2/slurm_report/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="/home/szhang2/AlchemyRepos/HunyuanVideo"
VENV_PATH="/home/szhang2/venvs/hunyuanvideo-py311/bin/activate"
MODEL_BASE="/projects/prjs1914/models/HunyuanVideo/ckpts"
OUTPUT_DIR="/projects/prjs1914/output/hunyuan_text_embeddings"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}"
export MODEL_BASE

python mywork/text_latent_extract/embedding_saving.py \
  --start-index 11 \
  --end-index 2179 \
  --skip-missing \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda