

cd /home/szhang2/AlchemyRepos/HunyuanVideo
source /home/szhang2/venvs/hunyuanvideo-py311/bin/activate

python mywork/vae_latent_extract/vae_latent_and_reconstruct.py \
  --process-all \
  --start-index 17 \
  --end-index 22 \
  --latent-only \
  --output-dir /projects/prjs1914/output/vae_latent \
  --vae-path /projects/prjs1914/models/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/vae \
  --device cuda