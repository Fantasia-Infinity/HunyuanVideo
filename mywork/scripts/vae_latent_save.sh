#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --job-name=vae_latent_0023_1023
#SBATCH --output=/home/szhang2/slurm_report/slurm-%j.out
#SBATCH --error=/home/szhang2/slurm_report/slurm-%j.err

set -euo pipefail

PROJECT_ROOT="/home/szhang2/AlchemyRepos/HunyuanVideo"
VENV_PATH="/home/szhang2/venvs/hunyuanvideo-py311/bin/activate"
VAE_PATH="/projects/prjs1914/models/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/vae"
OUTPUT_DIR="/projects/prjs1914/output/vae_latent"

cd "${PROJECT_ROOT}"
source "${VENV_PATH}"

python mywork/vae_latent_extract/vae_latent_and_reconstruct.py \
  --process-all \
  --start-index 23 \
  --end-index 1023 \
  --latent-only \
  --output-dir "${OUTPUT_DIR}" \
  --vae-path "${VAE_PATH}" \
  --device cuda