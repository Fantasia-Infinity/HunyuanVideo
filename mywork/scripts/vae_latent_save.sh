
#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/home/szhang2/AlchemyRepos/HunyuanVideo"
VENV_PATH="/home/szhang2/venvs/hunyuanvideo-py311/bin/activate"
VAE_PATH="/projects/prjs1914/models/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/vae"
OUTPUT_DIR="/projects/prjs1914/output/vae_latent"

srun \
  --account=vusr117232 \
  --partition=gpu_a100 \
  --gpus=1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=10:00:00 \
  bash -lc "
    set -euo pipefail
    cd ${PROJECT_ROOT}
    source ${VENV_PATH}
    python mywork/vae_latent_extract/vae_latent_and_reconstruct.py \
      --process-all \
      --start-index 23 \
      --end-index 1023 \
      --latent-only \
      --output-dir ${OUTPUT_DIR} \
      --vae-path ${VAE_PATH} \
      --device cuda
  "