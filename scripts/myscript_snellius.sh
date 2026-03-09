MODEL_BASE=/projects/prjs1914/models/HunyuanVideo/ckpts \
python sample_video.py \
  --model-base /projects/prjs1914/models/HunyuanVideo/ckpts \
  --dit-weight /projects/prjs1914/models/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
  --model-resolution 540p \
  --video-size 544 960 \
  --video-length 61 \
  --infer-steps 30 \
  --prompt "A person wears a colorful, patterned sweater and is seated, holding an open book with yellow pages. They flip through the pages with their right hand while the left hand supports the book from underneath." \
  --seed 42 \
  --embedded-cfg-scale 6.0 \
  --flow-shift 7.0 \
  --flow-reverse \
  --use-cpu-offload \
  --save-path /projects/prjs1914/DiffusionResults/hunyuantest

