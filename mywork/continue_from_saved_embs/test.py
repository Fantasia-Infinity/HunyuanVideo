import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL_BASE = "/projects/prjs1914/models/HunyuanVideo/ckpts"
DEFAULT_DIT_WEIGHT = "/projects/prjs1914/models/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
DEFAULT_TEXT_ARTIFACTS = "/projects/prjs1914/output/hunyuan_text_embeddings/0001_fw/text_encoder_artifacts.pt"
DEFAULT_VIDEO_LATENT = "/projects/prjs1914/output/vae_latent_540p/0001_fw_16by9_960x544_crop_540p/0001_fw_16by9_960x544_crop_540p_latent.pt"
DEFAULT_OUTPUT_DIR = "/projects/prjs1914/output/continue_from_saved_embs"


def _default_device():
	return "cuda" if torch.cuda.is_available() else "cpu"


def _load_payload(path):
	return torch.load(path, map_location="cpu")


def _ensure_pipeline_inputs(text_payload):
	pipeline_inputs = text_payload.get("pipeline_inputs")
	if pipeline_inputs is None:
		raise ValueError(
			"The text artifact file does not contain `pipeline_inputs`. Re-run the text save script after the pipeline-ready payload patch."
		)
	return pipeline_inputs


def _load_shared_negative_conditions(text_payload, pipeline_inputs):
	shared_info = pipeline_inputs.get("negative_prompt_source") or text_payload.get(
		"shared_negative_prompt_artifacts"
	)
	if shared_info is None:
		return None
	tensor_path = shared_info.get("tensor_output_path")
	if tensor_path is None:
		return None
	shared_payload = _load_payload(tensor_path)
	return {
		"negative_prompt": shared_payload["negative_prompt"],
		"llm": shared_payload["llm"],
		"clipL": shared_payload["clipL"],
	}


def _load_saved_conditions(text_artifacts_path, guidance_scale):
	text_payload = _load_payload(text_artifacts_path)
	pipeline_inputs = _ensure_pipeline_inputs(text_payload)
	shared_negative = _load_shared_negative_conditions(text_payload, pipeline_inputs)
	negative_prompt = pipeline_inputs.get("negative_prompt")
	negative_llm = None
	negative_clip = None
	if shared_negative is not None:
		negative_prompt = shared_negative["negative_prompt"]
		negative_llm = shared_negative["llm"]
		negative_clip = shared_negative["clipL"]
	else:
		negative_llm = pipeline_inputs.get("llm", {}).get("negative")
		negative_clip = pipeline_inputs.get("clipL", {}).get("negative")

	if guidance_scale > 1.0:
		if pipeline_inputs.get("cfg") is not None:
			prompt_embeds = pipeline_inputs["cfg"]["prompt_embeds"]
			attention_mask = pipeline_inputs["cfg"]["attention_mask"]
			prompt_embeds_2 = pipeline_inputs["cfg"]["prompt_embeds_2"]
		else:
			if negative_llm is None or negative_clip is None:
				raise ValueError(
					"Guidance requires saved negative prompt conditions, but none were found in the text artifact payload."
				)
			prompt_embeds = torch.cat(
				[
					negative_llm["prompt_embeds"],
					pipeline_inputs["llm"]["positive"]["prompt_embeds"],
				],
				dim=0,
			)
			positive_attention_mask = pipeline_inputs["llm"]["positive"].get("attention_mask")
			negative_attention_mask = negative_llm.get("attention_mask")
			attention_mask = None
			if positive_attention_mask is not None and negative_attention_mask is not None:
				attention_mask = torch.cat(
					[negative_attention_mask, positive_attention_mask],
					dim=0,
				)
			prompt_embeds_2 = torch.cat(
				[
					negative_clip["prompt_embeds"],
					pipeline_inputs["clipL"]["positive"]["prompt_embeds"],
				],
				dim=0,
			)
	else:
		prompt_embeds = pipeline_inputs["llm"]["positive"]["prompt_embeds"]
		attention_mask = pipeline_inputs["llm"]["positive"]["attention_mask"]
		prompt_embeds_2 = pipeline_inputs["clipL"]["positive"]["prompt_embeds"]

	return {
		"prompt": pipeline_inputs["prompt"],
		"negative_prompt": negative_prompt,
		"data_type": pipeline_inputs["data_type"],
		"prompt_embeds": prompt_embeds,
		"attention_mask": attention_mask,
		"prompt_embeds_2": prompt_embeds_2,
	}


def _load_saved_video_latents(video_latent_path, latent_key="latent"):
	latent_payload = _load_payload(video_latent_path)
	if latent_key not in latent_payload:
		raise KeyError(f"Missing latent key `{latent_key}` in {video_latent_path}")
	prepared_shape = latent_payload.get("prepared_video_shape")
	if prepared_shape is None or len(prepared_shape) != 5:
		raise ValueError(
			f"Saved latent payload does not have a valid `prepared_video_shape`: {prepared_shape}"
		)
	return {
		"latents": latent_payload[latent_key],
		"prepared_video_shape": prepared_shape,
		"fps": latent_payload.get("fps", 24),
		"input_video_path": latent_payload.get("input_video_path"),
		"latent_path": str(Path(video_latent_path).resolve()),
	}


def _import_hunyuan_runtime(model_base):
	os.environ["MODEL_BASE"] = model_base
	from hyvideo.config import parse_args
	from hyvideo.constants import PRECISION_TO_TYPE
	from hyvideo.diffusion.pipelines.pipeline_hunyuan_video import (
		rescale_noise_cfg,
		retrieve_timesteps,
	)
	from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
	from hyvideo.inference import HunyuanVideoSampler
	from hyvideo.utils.file_utils import save_videos_grid

	return {
		"parse_args": parse_args,
		"PRECISION_TO_TYPE": PRECISION_TO_TYPE,
		"rescale_noise_cfg": rescale_noise_cfg,
		"retrieve_timesteps": retrieve_timesteps,
		"FlowMatchDiscreteScheduler": FlowMatchDiscreteScheduler,
		"HunyuanVideoSampler": HunyuanVideoSampler,
		"save_videos_grid": save_videos_grid,
	}


def _build_sampler(
	model_base,
	dit_weight,
	model_resolution,
	device,
	precision,
	vae_precision,
	text_encoder_precision,
	text_encoder_precision_2,
	flow_shift,
):
	runtime = _import_hunyuan_runtime(model_base)
	args = runtime["parse_args"](
		[
			"--model-base",
			model_base,
			"--dit-weight",
			dit_weight,
			"--model-resolution",
			model_resolution,
			"--precision",
			precision,
			"--vae-precision",
			vae_precision,
			"--text-encoder-precision",
			text_encoder_precision,
			"--text-encoder-precision-2",
			text_encoder_precision_2,
			"--flow-shift",
			str(flow_shift),
			"--reproduce",
		]
	)
	sampler = runtime["HunyuanVideoSampler"].from_pretrained(
		Path(model_base),
		args=args,
		device=device,
	)
	return sampler, runtime


@torch.no_grad()
def continue_pipeline_from_saved_conditions(
	text_artifacts_path,
	video_latent_path,
	output_dir=DEFAULT_OUTPUT_DIR,
	model_base=DEFAULT_MODEL_BASE,
	dit_weight=DEFAULT_DIT_WEIGHT,
	model_resolution="540p",
	device=None,
	precision="bf16",
	vae_precision="fp16",
	text_encoder_precision="fp16",
	text_encoder_precision_2="fp16",
	latent_key="latent",
	infer_steps=50,
	guidance_scale=6.0,
	flow_shift=7.0,
	guidance_rescale=0.0,
	embedded_guidance_scale=6.0,
	enable_tiling=True,
):
	device = device or _default_device()
	sampler, runtime = _build_sampler(
		model_base=model_base,
		dit_weight=dit_weight,
		model_resolution=model_resolution,
		device=device,
		precision=precision,
		vae_precision=vae_precision,
		text_encoder_precision=text_encoder_precision,
		text_encoder_precision_2=text_encoder_precision_2,
		flow_shift=flow_shift,
	)

	conditions = _load_saved_conditions(text_artifacts_path, guidance_scale=guidance_scale)
	video_state = _load_saved_video_latents(video_latent_path, latent_key=latent_key)

	pipeline = sampler.pipeline
	pipe_device = pipeline._execution_device
	prepared_shape = video_state["prepared_video_shape"]
	batch_size, _, video_length, height, width = prepared_shape
	if batch_size != 1:
		raise ValueError(f"Only batch size 1 is supported for saved latent resume, got {batch_size}.")

	latents = video_state["latents"].to(
		device=pipe_device,
		dtype=runtime["PRECISION_TO_TYPE"][sampler.args.precision],
	)
	prompt_embeds = conditions["prompt_embeds"].to(
		device=pipe_device,
		dtype=pipeline.transformer.dtype,
	)
	attention_mask = conditions["attention_mask"]
	if attention_mask is not None:
		attention_mask = attention_mask.to(device=pipe_device)
	prompt_embeds_2 = conditions["prompt_embeds_2"].to(
		device=pipe_device,
		dtype=pipeline.transformer.dtype,
	)

	pipeline.scheduler = runtime["FlowMatchDiscreteScheduler"](
		shift=flow_shift,
		reverse=sampler.args.flow_reverse,
		solver=sampler.args.flow_solver,
	)
	freqs_cos, freqs_sin = sampler.get_rotary_pos_embed(video_length, height, width)
	n_tokens = freqs_cos.shape[0]
	timesteps, num_inference_steps = runtime["retrieve_timesteps"](
		pipeline.scheduler,
		num_inference_steps=infer_steps,
		device=pipe_device,
		n_tokens=n_tokens,
	)

	target_dtype = runtime["PRECISION_TO_TYPE"][sampler.args.precision]
	autocast_enabled = (
		target_dtype != torch.float32
	) and not sampler.args.disable_autocast
	vae_dtype = runtime["PRECISION_TO_TYPE"][sampler.args.vae_precision]
	vae_autocast_enabled = (
		vae_dtype != torch.float32
	) and not sampler.args.disable_autocast

	start_time = time.time()
	num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
	pipeline._guidance_scale = guidance_scale
	pipeline._guidance_rescale = guidance_rescale

	with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
		for step_index, timestep in enumerate(timesteps):
			latent_model_input = (
				torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
			)
			latent_model_input = pipeline.scheduler.scale_model_input(
				latent_model_input, timestep
			)

			t_expand = timestep.repeat(latent_model_input.shape[0])
			guidance_expand = (
				torch.tensor(
					[embedded_guidance_scale] * latent_model_input.shape[0],
					dtype=torch.float32,
					device=pipe_device,
				).to(target_dtype)
				* 1000.0
				if embedded_guidance_scale is not None
				else None
			)

			with torch.autocast(
				device_type="cuda",
				dtype=target_dtype,
				enabled=autocast_enabled and str(pipe_device).startswith("cuda"),
			):
				noise_pred = pipeline.transformer(
					latent_model_input,
					t_expand,
					text_states=prompt_embeds,
					text_mask=attention_mask,
					text_states_2=prompt_embeds_2,
					freqs_cos=freqs_cos,
					freqs_sin=freqs_sin,
					guidance=guidance_expand,
					return_dict=True,
				)["x"]

			if guidance_scale > 1.0:
				noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
				noise_pred = noise_pred_uncond + guidance_scale * (
					noise_pred_text - noise_pred_uncond
				)
				if guidance_rescale > 0.0:
					noise_pred = runtime["rescale_noise_cfg"](
						noise_pred,
						noise_pred_text,
						guidance_rescale=guidance_rescale,
					)

			latents = pipeline.scheduler.step(
				noise_pred,
				timestep,
				latents,
				return_dict=False,
			)[0]

			if step_index == len(timesteps) - 1 or (
				(step_index + 1) > num_warmup_steps
				and (step_index + 1) % pipeline.scheduler.order == 0
			):
				progress_bar.update()

	if hasattr(pipeline.vae.config, "shift_factor") and pipeline.vae.config.shift_factor:
		decode_latents = latents / pipeline.vae.config.scaling_factor + pipeline.vae.config.shift_factor
	else:
		decode_latents = latents / pipeline.vae.config.scaling_factor

	with torch.autocast(
		device_type="cuda",
		dtype=vae_dtype,
		enabled=vae_autocast_enabled and str(pipe_device).startswith("cuda"),
	):
		if enable_tiling:
			pipeline.vae.enable_tiling()
		decoded_video = pipeline.vae.decode(
			decode_latents,
			return_dict=False,
		)[0]

	decoded_video = (decoded_video / 2 + 0.5).clamp(0, 1).cpu().float()

	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	sample_name = f"{Path(text_artifacts_path).stem}__{Path(video_latent_path).stem}"
	video_output_path = output_root / f"{sample_name}_continued.mp4"
	metadata_output_path = output_root / f"{sample_name}_continued.json"
	runtime["save_videos_grid"](
		decoded_video,
		str(video_output_path),
		rescale=False,
		fps=video_state["fps"],
	)

	metadata = {
		"text_artifacts_path": str(Path(text_artifacts_path).resolve()),
		"video_latent_path": video_state["latent_path"],
		"input_video_path": video_state["input_video_path"],
		"prompt": conditions["prompt"],
		"negative_prompt": conditions["negative_prompt"],
		"data_type": conditions["data_type"],
		"video_length": video_length,
		"height": height,
		"width": width,
		"fps": video_state["fps"],
		"guidance_scale": guidance_scale,
		"guidance_rescale": guidance_rescale,
		"embedded_guidance_scale": embedded_guidance_scale,
		"infer_steps": infer_steps,
		"flow_shift": flow_shift,
		"latent_key": latent_key,
		"latents_shape": list(video_state["latents"].shape),
		"prompt_embeds_shape": list(conditions["prompt_embeds"].shape),
		"prompt_embeds_2_shape": list(conditions["prompt_embeds_2"].shape),
		"attention_mask_shape": list(conditions["attention_mask"].shape) if conditions["attention_mask"] is not None else None,
		"output_video_path": str(video_output_path),
		"elapsed_seconds": time.time() - start_time,
	}
	metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
	return metadata


def parse_args():
	parser = argparse.ArgumentParser(
		description="Continue the HunyuanVideo diffusion pipeline from saved text embeddings and saved VAE latents."
	)
	parser.add_argument("--text-artifacts", default=DEFAULT_TEXT_ARTIFACTS, help="Path to the saved text encoder artifact .pt file.")
	parser.add_argument("--video-latent", default=DEFAULT_VIDEO_LATENT, help="Path to the saved VAE latent .pt file.")
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory used to save the generated video and metadata.")
	parser.add_argument("--model-base", default=DEFAULT_MODEL_BASE, help="MODEL_BASE used to locate HunyuanVideo checkpoints.")
	parser.add_argument("--dit-weight", default=DEFAULT_DIT_WEIGHT, help="Path to the HunyuanVideo DiT checkpoint.")
	parser.add_argument("--model-resolution", default="540p", choices=["540p", "720p"], help="Model resolution variant used for the DiT checkpoint.")
	parser.add_argument("--device", default=None, help="Execution device, for example cuda or cpu.")
	parser.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"], help="Transformer precision.")
	parser.add_argument("--vae-precision", default="fp16", choices=["fp16", "bf16", "fp32"], help="VAE precision.")
	parser.add_argument("--text-encoder-precision", default="fp16", choices=["fp16", "bf16", "fp32"], help="Precision used when constructing the primary text encoder scaffold.")
	parser.add_argument("--text-encoder-precision-2", default="fp16", choices=["fp16", "bf16", "fp32"], help="Precision used when constructing the CLIP text encoder scaffold.")
	parser.add_argument("--latent-key", default="latent", choices=["latent", "posterior_mean"], help="Which saved latent tensor to use as the initial diffusion latent.")
	parser.add_argument("--infer-steps", type=int, default=50, help="Number of diffusion denoising steps.")
	parser.add_argument("--guidance-scale", type=float, default=6.0, help="Classifier-free guidance scale.")
	parser.add_argument("--guidance-rescale", type=float, default=0.0, help="Guidance rescale factor.")
	parser.add_argument("--embedded-guidance-scale", type=float, default=6.0, help="Embedded guidance scale passed into the transformer modulation branch.")
	parser.add_argument("--flow-shift", type=float, default=7.0, help="Flow matching shift factor.")
	parser.add_argument("--disable-tiling", action="store_true", help="Disable VAE tiling during decode.")
	return parser.parse_args()


def main():
	args = parse_args()
	result = continue_pipeline_from_saved_conditions(
		text_artifacts_path=args.text_artifacts,
		video_latent_path=args.video_latent,
		output_dir=args.output_dir,
		model_base=args.model_base,
		dit_weight=args.dit_weight,
		model_resolution=args.model_resolution,
		device=args.device,
		precision=args.precision,
		vae_precision=args.vae_precision,
		text_encoder_precision=args.text_encoder_precision,
		text_encoder_precision_2=args.text_encoder_precision_2,
		latent_key=args.latent_key,
		infer_steps=args.infer_steps,
		guidance_scale=args.guidance_scale,
		flow_shift=args.flow_shift,
		guidance_rescale=args.guidance_rescale,
		embedded_guidance_scale=args.embedded_guidance_scale,
		enable_tiling=not args.disable_tiling,
	)
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
