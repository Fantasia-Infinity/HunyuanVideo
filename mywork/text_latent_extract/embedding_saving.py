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


def _default_model_base():
	project_ckpts = Path("/projects/prjs1914/models/HunyuanVideo/ckpts")
	if project_ckpts.exists():
		return str(project_ckpts)
	return str(REPO_ROOT / "ckpts")


DEFAULT_MODEL_BASE = _default_model_base()
os.environ.setdefault("MODEL_BASE", DEFAULT_MODEL_BASE)

from hyvideo.constants import NEGATIVE_PROMPT, PROMPT_TEMPLATE
from hyvideo.text_encoder import TextEncoder


EXAMPLE_INPUT_FILE = "/projects/prjs1914/input/qwen_describe_shorten/0001_fw.txt"
DEFAULT_INPUT_TEMPLATE = "/projects/prjs1914/input/qwen_describe_shorten/{index:04d}_fw.txt"
DEFAULT_OUTPUT_DIR = "/projects/prjs1914/output/hunyuan_text_embeddings"
DEFAULT_PROMPT_TEMPLATE = "dit-llm-encode"
DEFAULT_PROMPT_TEMPLATE_VIDEO = "dit-llm-encode-video"
SHARED_NEGATIVE_FILENAME = "shared_negative_prompt_artifacts.pt"
SHARED_NEGATIVE_SUMMARY_FILENAME = "shared_negative_prompt_artifacts_summary.json"


DEFAULT_TEXT_ENCODER_PRECISION = "fp16"
DEFAULT_TEXT_LEN = 30



def _default_device():
	return "cuda" if torch.cuda.is_available() else "cpu"


def _default_precision(device):
	return "fp16" if str(device).startswith("cuda") else "fp32"


def _read_prompt_text(input_file):
	input_path = Path(input_file)
	prompt = input_path.read_text(encoding="utf-8").strip()
	if not prompt:
		raise ValueError(f"Prompt file is empty: {input_path}")
	return prompt


def _prompt_template_for_data_type(text_encoder, data_type):
	if not text_encoder.use_template:
		return None
	if data_type == "image":
		return text_encoder.prompt_template
	if data_type == "video":
		return text_encoder.prompt_template_video
	raise ValueError(f"Unsupported data_type: {data_type}")


def _applied_prompt_text(text_encoder, prompt, data_type):
	prompt_template = _prompt_template_for_data_type(text_encoder, data_type)
	if prompt_template is None:
		return prompt
	return text_encoder.apply_text_to_template(prompt, prompt_template["template"])


def _crop_start(text_encoder, data_type):
	prompt_template = _prompt_template_for_data_type(text_encoder, data_type)
	if prompt_template is None:
		return 0
	return int(prompt_template.get("crop_start", 0))


def _crop_sequence_tensor(tensor, crop_start):
	if tensor is None or crop_start <= 0 or tensor.ndim != 3:
		return tensor
	return tensor[:, crop_start:]


def _tensor_to_cpu(tensor):
	if tensor is None:
		return None
	return tensor.detach().cpu()


def _summarize_value(value):
	if torch.is_tensor(value):
		return {
			"shape": list(value.shape),
			"dtype": str(value.dtype),
		}
	if isinstance(value, (list, tuple)):
		if value and all(torch.is_tensor(item) for item in value):
			return [
				{
					"shape": list(item.shape),
					"dtype": str(item.dtype),
				}
				for item in value
			]
		return value
	if isinstance(value, dict):
		return {key: _summarize_value(item) for key, item in value.items()}
	return value


def collect_text_encoder_artifacts(text_encoder, prompt, data_type="video", device=None):
	device = device or text_encoder.device
	applied_text = _applied_prompt_text(text_encoder, prompt, data_type)
	crop_start = _crop_start(text_encoder, data_type)
	batch_encoding = text_encoder.text2tokens(prompt, data_type=data_type)
	attention_mask = batch_encoding.get("attention_mask")
	model_outputs = text_encoder.model(
		input_ids=batch_encoding["input_ids"].to(device),
		attention_mask=attention_mask.to(device) if attention_mask is not None else None,
		output_hidden_states=True,
	)

	selected_hidden_state = None
	if text_encoder.hidden_state_skip_layer is not None:
		selected_hidden_state = model_outputs.hidden_states[
			-(text_encoder.hidden_state_skip_layer + 1)
		]
		if text_encoder.hidden_state_skip_layer > 0 and text_encoder.apply_final_norm:
			selected_hidden_state = text_encoder.model.final_layer_norm(selected_hidden_state)
	else:
		selected_hidden_state = model_outputs[text_encoder.output_key]

	last_hidden_state = getattr(model_outputs, "last_hidden_state", None)
	pooler_output = getattr(model_outputs, "pooler_output", None)
	hidden_states = tuple(model_outputs.hidden_states) if model_outputs.hidden_states is not None else tuple()

	selected_hidden_state_cropped = _crop_sequence_tensor(selected_hidden_state, crop_start)
	last_hidden_state_cropped = _crop_sequence_tensor(last_hidden_state, crop_start)
	hidden_states_cropped = tuple(_crop_sequence_tensor(item, crop_start) for item in hidden_states)
	attention_mask_cropped = (
		attention_mask[:, crop_start:] if attention_mask is not None and crop_start > 0 else attention_mask
	)

	input_ids_cpu = batch_encoding["input_ids"].detach().cpu()
	attention_mask_cpu = attention_mask.detach().cpu() if attention_mask is not None else None
	attention_mask_cropped_cpu = (
		attention_mask_cropped.detach().cpu() if attention_mask_cropped is not None else None
	)

	return {
		"text_encoder_type": text_encoder.text_encoder_type,
		"tokenizer_type": text_encoder.tokenizer_type,
		"model_path": str(text_encoder.model_path),
		"tokenizer_path": str(text_encoder.tokenizer_path),
		"dtype": str(text_encoder.dtype),
		"device": str(device),
		"output_key": text_encoder.output_key,
		"data_type": data_type,
		"use_template": text_encoder.use_template,
		"applied_text": applied_text,
		"crop_start": crop_start,
		"input_ids": input_ids_cpu,
		"attention_mask": attention_mask_cpu,
		"attention_mask_cropped": attention_mask_cropped_cpu,
		"tokens": text_encoder.tokenizer.convert_ids_to_tokens(input_ids_cpu[0].tolist()),
		"decoded_text": text_encoder.tokenizer.batch_decode(input_ids_cpu, skip_special_tokens=False),
		"selected_hidden_state": _tensor_to_cpu(selected_hidden_state_cropped),
		"selected_hidden_state_pre_crop": _tensor_to_cpu(selected_hidden_state),
		"last_hidden_state": _tensor_to_cpu(last_hidden_state_cropped),
		"last_hidden_state_pre_crop": _tensor_to_cpu(last_hidden_state),
		"pooler_output": _tensor_to_cpu(pooler_output),
		"hidden_states": tuple(_tensor_to_cpu(item) for item in hidden_states_cropped),
		"hidden_states_pre_crop": tuple(_tensor_to_cpu(item) for item in hidden_states),
	}


def _encode_prompt_for_pipeline(text_encoder, prompt, data_type="video", device=None):
	device = device or text_encoder.device
	text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)
	outputs = text_encoder.encode(text_inputs, data_type=data_type, device=device)
	return {
		"prompt_text": prompt,
		"applied_text": _applied_prompt_text(text_encoder, prompt, data_type),
		"input_ids": text_inputs["input_ids"].detach().cpu(),
		"attention_mask_pre_crop": (
			text_inputs["attention_mask"].detach().cpu()
			if text_inputs.get("attention_mask") is not None
			else None
		),
		"prompt_embeds": _tensor_to_cpu(outputs.hidden_state),
		"attention_mask": _tensor_to_cpu(outputs.attention_mask),
	}


def build_pipeline_ready_text_conditions(
	encoders,
	prompt,
	negative_prompt,
	data_type="video",
	device=None,
):
	llm_encoder = encoders["llm"]
	clip_encoder = encoders["clipL"]

	llm_positive = _encode_prompt_for_pipeline(
		llm_encoder,
		prompt=prompt,
		data_type=data_type,
		device=device or llm_encoder.device,
	)
	llm_negative = _encode_prompt_for_pipeline(
		llm_encoder,
		prompt=negative_prompt,
		data_type=data_type,
		device=device or llm_encoder.device,
	)
	clip_positive = _encode_prompt_for_pipeline(
		clip_encoder,
		prompt=prompt,
		data_type=data_type,
		device=device or clip_encoder.device,
	)
	clip_negative = _encode_prompt_for_pipeline(
		clip_encoder,
		prompt=negative_prompt,
		data_type=data_type,
		device=device or clip_encoder.device,
	)

	cfg_prompt_embeds = torch.cat(
		[llm_negative["prompt_embeds"], llm_positive["prompt_embeds"]], dim=0
	)
	cfg_attention_mask = None
	if llm_positive["attention_mask"] is not None:
		cfg_attention_mask = torch.cat(
			[llm_negative["attention_mask"], llm_positive["attention_mask"]], dim=0
		)
	cfg_prompt_embeds_2 = torch.cat(
		[clip_negative["prompt_embeds"], clip_positive["prompt_embeds"]], dim=0
	)

	return {
		"prompt": prompt,
		"negative_prompt": negative_prompt,
		"data_type": data_type,
		"llm": {
			"positive": llm_positive,
			"negative": llm_negative,
		},
		"clipL": {
			"positive": clip_positive,
			"negative": clip_negative,
		},
		"cfg": {
			"prompt_embeds": cfg_prompt_embeds,
			"attention_mask": cfg_attention_mask,
			"prompt_embeds_2": cfg_prompt_embeds_2,
		},
	}


def _shared_negative_output_paths(output_root):
	output_root = Path(output_root)
	return (
		output_root / SHARED_NEGATIVE_FILENAME,
		output_root / SHARED_NEGATIVE_SUMMARY_FILENAME,
	)


def save_shared_negative_prompt_artifacts(
	encoders,
	negative_prompt,
	output_dir,
	data_type="video",
	device=None,
):
	tensor_output_path, summary_output_path = _shared_negative_output_paths(output_dir)
	payload = {
		"negative_prompt": negative_prompt,
		"data_type": data_type,
		"llm": _encode_prompt_for_pipeline(
			encoders["llm"],
			prompt=negative_prompt,
			data_type=data_type,
			device=device or encoders["llm"].device,
		),
		"clipL": _encode_prompt_for_pipeline(
			encoders["clipL"],
			prompt=negative_prompt,
			data_type=data_type,
			device=device or encoders["clipL"].device,
		),
	}
	tensor_output_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(payload, tensor_output_path)
	summary_output_path.write_text(
		json.dumps(_summarize_value(payload), indent=2),
		encoding="utf-8",
	)
	return {
		"tensor_output_path": str(tensor_output_path),
		"summary_output_path": str(summary_output_path),
	}


def build_hunyuan_text_encoders(
	device=None,
	text_encoder_precision=None,
	text_encoder_precision_2=None,
	text_len=256,
	text_len_2=77,
	prompt_template=DEFAULT_PROMPT_TEMPLATE,
	prompt_template_video=DEFAULT_PROMPT_TEMPLATE_VIDEO,
	hidden_state_skip_layer=2,
	apply_final_norm=False,
):
	device = device or _default_device()
	text_encoder_precision = text_encoder_precision or _default_precision(device)
	text_encoder_precision_2 = text_encoder_precision_2 or _default_precision(device)

	crop_start = 0
	if prompt_template_video is not None:
		crop_start = PROMPT_TEMPLATE[prompt_template_video].get("crop_start", 0)
	elif prompt_template is not None:
		crop_start = PROMPT_TEMPLATE[prompt_template].get("crop_start", 0)

	llm_encoder = TextEncoder(
		text_encoder_type="llm",
		max_length=text_len + crop_start,
		text_encoder_precision=text_encoder_precision,
		tokenizer_type="llm",
		prompt_template=PROMPT_TEMPLATE[prompt_template] if prompt_template is not None else None,
		prompt_template_video=(
			PROMPT_TEMPLATE[prompt_template_video] if prompt_template_video is not None else None
		),
		hidden_state_skip_layer=hidden_state_skip_layer,
		apply_final_norm=apply_final_norm,
		reproduce=True,
		device=device,
	)
	clip_encoder = TextEncoder(
		text_encoder_type="clipL",
		max_length=text_len_2,
		text_encoder_precision=text_encoder_precision_2,
		tokenizer_type="clipL",
		reproduce=True,
		device=device,
	)
	return {
		"llm": llm_encoder,
		"clipL": clip_encoder,
	}


def save_text_encoder_artifacts_from_file(
	input_file,
	output_dir=DEFAULT_OUTPUT_DIR,
	data_type="video",
	negative_prompt=NEGATIVE_PROMPT,
	save_encoder_artifacts=False,
	device=None,
	text_encoder_precision=None,
	text_encoder_precision_2=None,
	text_len=256,
	text_len_2=77,
	prompt_template=DEFAULT_PROMPT_TEMPLATE,
	prompt_template_video=DEFAULT_PROMPT_TEMPLATE_VIDEO,
	hidden_state_skip_layer=2,
	apply_final_norm=False,
):
	start_time = time.perf_counter()
	prompt = _read_prompt_text(input_file)
	encoders = build_hunyuan_text_encoders(
		device=device,
		text_encoder_precision=text_encoder_precision,
		text_encoder_precision_2=text_encoder_precision_2,
		text_len=text_len,
		text_len_2=text_len_2,
		prompt_template=prompt_template,
		prompt_template_video=prompt_template_video,
		hidden_state_skip_layer=hidden_state_skip_layer,
		apply_final_norm=apply_final_norm,
	)

	output_root = Path(output_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	sample_output_dir = output_root / Path(input_file).stem
	sample_output_dir.mkdir(parents=True, exist_ok=True)
	shared_negative_paths = save_shared_negative_prompt_artifacts(
		encoders=encoders,
		negative_prompt=negative_prompt,
		output_dir=output_root,
		data_type=data_type,
		device=device,
	)

	payload = {
		"input_file": str(Path(input_file).resolve()),
		"prompt": prompt,
		"negative_prompt": negative_prompt,
		"data_type": data_type,
		"shared_negative_prompt_artifacts": shared_negative_paths,
	}
	if save_encoder_artifacts:
		payload["encoders"] = {}
		for encoder_name, text_encoder in encoders.items():
			payload["encoders"][encoder_name] = collect_text_encoder_artifacts(
				text_encoder=text_encoder,
				prompt=prompt,
				data_type=data_type,
				device=device or text_encoder.device,
			)
	positive_pipeline_inputs = build_pipeline_ready_text_conditions(
		encoders=encoders,
		prompt=prompt,
		negative_prompt=negative_prompt,
		data_type=data_type,
		device=device,
	)
	payload["pipeline_inputs"] = {
		"prompt": positive_pipeline_inputs["prompt"],
		"negative_prompt": positive_pipeline_inputs["negative_prompt"],
		"data_type": positive_pipeline_inputs["data_type"],
		"llm": {
			"positive": positive_pipeline_inputs["llm"]["positive"],
		},
		"clipL": {
			"positive": positive_pipeline_inputs["clipL"]["positive"],
		},
		"negative_prompt_source": shared_negative_paths,
	}

	tensor_output_path = sample_output_dir / "text_encoder_artifacts.pt"
	summary_output_path = sample_output_dir / "text_encoder_artifacts_summary.json"
	torch.save(payload, tensor_output_path)
	summary_output_path.write_text(
		json.dumps(_summarize_value(payload), indent=2),
		encoding="utf-8",
	)
	elapsed_seconds = time.perf_counter() - start_time

	return {
		"input_file": payload["input_file"],
		"output_dir": str(sample_output_dir),
		"tensor_output_path": str(tensor_output_path),
		"summary_output_path": str(summary_output_path),
		"encoders": list(payload.get("encoders", {}).keys()),
		"save_encoder_artifacts": save_encoder_artifacts,
		"elapsed_seconds": elapsed_seconds,
	}


def save_text_encoder_artifacts_for_range(
	start_index,
	end_index,
	input_template=DEFAULT_INPUT_TEMPLATE,
	skip_missing=True,
	**kwargs,
):
	batch_start_time = time.perf_counter()
	results = []
	skipped = []
	for video_index in range(start_index, end_index + 1):
		input_file = input_template.format(index=video_index)
		input_path = Path(input_file)
		if not input_path.exists():
			if skip_missing:
				skipped.append(str(input_path))
				continue
			raise FileNotFoundError(f"Prompt file not found: {input_path}")

		result = save_text_encoder_artifacts_from_file(
			input_file=str(input_path),
			**kwargs,
		)
		result["video_index"] = video_index
		results.append(result)
	total_elapsed_seconds = time.perf_counter() - batch_start_time
	average_elapsed_seconds = (
		total_elapsed_seconds / len(results) if results else 0.0
	)

	return {
		"start_index": start_index,
		"end_index": end_index,
		"input_template": input_template,
		"processed": len(results),
		"skipped": len(skipped),
		"total_elapsed_seconds": total_elapsed_seconds,
		"average_elapsed_seconds": average_elapsed_seconds,
		"results": results,
		"skipped_files": skipped,
	}


def parse_args():
	parser = argparse.ArgumentParser(
		description="Save HunyuanVideo text encoder embeddings and related artifacts from a prompt text file."
	)
	parser.add_argument("--input-file", default=EXAMPLE_INPUT_FILE, help="Path to the input prompt text file.")
	parser.add_argument("--input-template", default=DEFAULT_INPUT_TEMPLATE, help="Template used for batch mode. Example: /path/{index:04d}_fw.txt")
	parser.add_argument("--start-index", type=int, default=None, help="Start video index for batch mode.")
	parser.add_argument("--end-index", type=int, default=None, help="End video index for batch mode.")
	parser.add_argument("--skip-missing", action="store_true", help="Skip missing prompt files in batch mode instead of failing.")
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory used to save the extracted artifacts.")
	parser.add_argument("--data-type", default="video", choices=["image", "video"], help="Whether to apply the image or video prompt template for the main LLM encoder.")
	parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT, help="Negative prompt used to build pipeline-ready unconditional text conditions.")
	parser.add_argument("--save-encoder-artifacts", action="store_true", help="Also save full per-encoder analysis artifacts, including per-layer hidden states. Disabled by default to keep files smaller and pipeline-focused.")
	parser.add_argument("--device", default=None, help="Execution device, for example cuda or cpu.")
	parser.add_argument("--text-encoder-precision", default=DEFAULT_TEXT_ENCODER_PRECISION, choices=[None, "fp16", "bf16", "fp32"], help="Precision for the main LLM text encoder. Defaults to fp16 on CUDA and fp32 on CPU.")
	parser.add_argument("--text-encoder-precision-2", default=DEFAULT_TEXT_ENCODER_PRECISION, choices=[None, "fp16", "bf16", "fp32"], help="Precision for the CLIP text encoder. Defaults to fp16 on CUDA and fp32 on CPU.")
	parser.add_argument("--text-len", type=int, default=DEFAULT_TEXT_LEN, help="Target prompt token length for the main LLM encoder after template cropping.")
	parser.add_argument("--text-len-2", type=int, default=DEFAULT_TEXT_LEN, help="Token length for the CLIP text encoder.")
	parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE, choices=list(PROMPT_TEMPLATE), help="Prompt template key for image-mode LLM encoding.")
	parser.add_argument("--prompt-template-video", default=DEFAULT_PROMPT_TEMPLATE_VIDEO, choices=list(PROMPT_TEMPLATE), help="Prompt template key for video-mode LLM encoding.")
	parser.add_argument("--hidden-state-skip-layer", type=int, default=2, help="Which intermediate LLM hidden state to expose as the selected embedding. 0 means the last layer.")
	parser.add_argument("--apply-final-norm", action="store_true", help="Apply the text encoder final norm when using an intermediate hidden state.")
	return parser.parse_args()


def main():
	args = parse_args()
	common_kwargs = {
		"output_dir": args.output_dir,
		"data_type": args.data_type,
		"negative_prompt": args.negative_prompt,
		"save_encoder_artifacts": args.save_encoder_artifacts,
		"device": args.device,
		"text_encoder_precision": args.text_encoder_precision,
		"text_encoder_precision_2": args.text_encoder_precision_2,
		"text_len": args.text_len,
		"text_len_2": args.text_len_2,
		"prompt_template": args.prompt_template,
		"prompt_template_video": args.prompt_template_video,
		"hidden_state_skip_layer": args.hidden_state_skip_layer,
		"apply_final_norm": args.apply_final_norm,
	}
	if args.start_index is not None or args.end_index is not None:
		if args.start_index is None or args.end_index is None:
			raise ValueError("Both --start-index and --end-index must be provided for batch mode.")
		result = save_text_encoder_artifacts_for_range(
			start_index=args.start_index,
			end_index=args.end_index,
			input_template=args.input_template,
			skip_missing=args.skip_missing,
			**common_kwargs,
		)
	else:
		result = save_text_encoder_artifacts_from_file(
			input_file=args.input_file,
			**common_kwargs,
		)
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()

