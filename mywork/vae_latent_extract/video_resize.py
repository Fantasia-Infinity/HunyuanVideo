import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np


sample_input = "/projects/prjs1914/input/rescaled_final/0001_fw.mp4"
output_dir = "/projects/prjs1914/input/rescaled_final_540p"

# HunyuanVideo's supported 540p presets. 544 is used instead of 540 so dimensions
# stay aligned with the model's latent downsampling requirements.
SUPPORTED_540P_SIZES = {
	"16:9": (960, 544),
	"9:16": (544, 960),
	"4:3": (832, 624),
	"3:4": (624, 832),
	"1:1": (720, 720),
}


def _closest_540p_size(width, height):
	aspect = width / height
	candidates = {
		key: size[0] / size[1] for key, size in SUPPORTED_540P_SIZES.items()
	}
	best_key = min(candidates, key=lambda key: abs(candidates[key] - aspect))
	return SUPPORTED_540P_SIZES[best_key], best_key


def _resize_with_crop(frame, target_width, target_height, interpolation):
	src_height, src_width = frame.shape[:2]
	scale = max(target_width / src_width, target_height / src_height)
	resized_width = int(round(src_width * scale))
	resized_height = int(round(src_height * scale))
	resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)

	x0 = max((resized_width - target_width) // 2, 0)
	y0 = max((resized_height - target_height) // 2, 0)
	return resized[y0:y0 + target_height, x0:x0 + target_width]


def _resize_with_pad(frame, target_width, target_height, interpolation):
	src_height, src_width = frame.shape[:2]
	scale = min(target_width / src_width, target_height / src_height)
	resized_width = int(round(src_width * scale))
	resized_height = int(round(src_height * scale))
	resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)

	canvas = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
	x0 = (target_width - resized_width) // 2
	y0 = (target_height - resized_height) // 2
	canvas[y0:y0 + resized_height, x0:x0 + resized_width] = resized
	return canvas


def _resize_with_stretch(frame, target_width, target_height, interpolation):
	return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def resize_frame_to_540p(frame, target_width, target_height, method="crop", interpolation=cv2.INTER_AREA):
	if method == "crop":
		return _resize_with_crop(frame, target_width, target_height, interpolation)
	if method == "pad":
		return _resize_with_pad(frame, target_width, target_height, interpolation)
	if method == "stretch":
		return _resize_with_stretch(frame, target_width, target_height, interpolation)
	raise ValueError(f"Unsupported resize method: {method}")


def resize_video_to_540p(input_video_path, output_dir, method="crop", output_name=None):
	"""Resize a video to the nearest HunyuanVideo 540p preset.

	Methods:
	- crop: keep aspect ratio, fill target canvas, then center crop.
	- pad: keep aspect ratio, fit inside target canvas, then pad borders.
	- stretch: ignore aspect ratio and resize directly to target size.
	"""
	input_video_path = Path(input_video_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	reader = imageio.get_reader(str(input_video_path))
	try:
		meta = reader.get_meta_data()
		fps = meta.get("fps", 24)
		first_frame = reader.get_data(0)
		src_height, src_width = first_frame.shape[:2]
		(target_width, target_height), preset_name = _closest_540p_size(src_width, src_height)

		frames = [
			resize_frame_to_540p(frame, target_width, target_height, method=method)
			for frame in reader
		]
	finally:
		reader.close()

	if not frames:
		raise ValueError(f"No frames were read from {input_video_path}")

	if output_name is None:
		suffix = f"_{preset_name.replace(':', 'by')}_{target_width}x{target_height}_{method}_540p"
		output_name = f"{input_video_path.stem}{suffix}{input_video_path.suffix}"
	output_path = output_dir / output_name
	imageio.mimsave(str(output_path), frames, fps=fps, macro_block_size=1)

	return {
		"input_video_path": str(input_video_path),
		"output_video_path": str(output_path),
		"source_width": src_width,
		"source_height": src_height,
		"target_width": target_width,
		"target_height": target_height,
		"preset_name": preset_name,
		"method": method,
		"fps": fps,
		"num_frames": len(frames),
	}


def parse_args():
	parser = argparse.ArgumentParser(description="Resize a video to a HunyuanVideo-compatible 540p preset.")
	parser.add_argument("--input", default=sample_input, help="Path to the input video.")
	parser.add_argument("--output-dir", default=output_dir, help="Directory where the resized video will be written.")
	parser.add_argument("--method", default="crop", choices=["crop", "pad", "stretch"], help="Resize strategy.")
	parser.add_argument("--output-name", default=None, help="Optional output file name.")
	return parser.parse_args()


def main():
	args = parse_args()
	info = resize_video_to_540p(
		input_video_path=args.input,
		output_dir=args.output_dir,
		method=args.method,
		output_name=args.output_name,
	)
	print(info)


if __name__ == "__main__":
	main()

