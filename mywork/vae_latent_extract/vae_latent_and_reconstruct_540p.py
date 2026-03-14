
import argparse
import json
from pathlib import Path
import sys
import time

import imageio
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hyvideo.utils.file_utils import safe_dir, save_videos_grid
from hyvideo.vae import load_vae


# DEFAULT_INPUT = "/projects/prjs1914/input/rescaled_final/0001_fw.mp4"
DEFAULT_INPUT = "/projects/prjs1914/input/rescaled_final_540p/0001_fw_16by9_960x544_crop_540p.mp4"
DEFAULT_OUTPUT_DIR = "/projects/prjs1914/output/vae_latent_540p"
DEFAULT_VAE_PATH = "/projects/prjs1914/models/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/vae"
DEFAULT_START_INDEX = 1
DEFAULT_END_INDEX = 2179


def video_index_to_path(video_index):
    return f"/projects/prjs1914/input/rescaled_final_540p/{video_index:04d}_fw_16by9_960x544_crop_540p.mp4"


def _valid_video_length(num_frames):
    if num_frames < 1:
        raise ValueError("Video must contain at least one frame.")
    if num_frames == 1:
        return 1
    return ((num_frames - 1) // 4) * 4 + 1


def _center_crop_bounds(length, multiple):
    cropped = length - (length % multiple)
    if cropped < multiple:
        raise ValueError(
            f"Dimension {length} is smaller than the required multiple {multiple}."
        )
    start = (length - cropped) // 2
    end = start + cropped
    return start, end


def load_video_tensor(video_path):
    video_path = Path(video_path)
    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps", 24)
        frames = []
        for frame in reader:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
            frames.append(frame_tensor)
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"No frames were read from {video_path}.")

    video = torch.stack(frames, dim=1).float()
    video = video / 127.5 - 1.0
    return video.unsqueeze(0), fps


def prepare_video_for_vae(video):
    _, _, num_frames, height, width = video.shape
    target_frames = _valid_video_length(num_frames)
    h_start, h_end = _center_crop_bounds(height, 8)
    w_start, w_end = _center_crop_bounds(width, 8)

    prepared = video[:, :, :target_frames, h_start:h_end, w_start:w_end]
    metadata = {
        "original_frames": num_frames,
        "used_frames": target_frames,
        "original_height": height,
        "original_width": width,
        "used_height": h_end - h_start,
        "used_width": w_end - w_start,
        "crop_top": h_start,
        "crop_left": w_start,
    }
    return prepared, metadata


def initialize_vae(
    vae_path=DEFAULT_VAE_PATH,
    vae_precision="fp16",
    device=None,
    enable_tiling=True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vae, resolved_vae_path, s_ratio, t_ratio = load_vae(
        vae_precision=vae_precision,
        vae_path=vae_path,
        device=device,
    )
    if enable_tiling:
        vae.enable_tiling()
    return vae, resolved_vae_path, s_ratio, t_ratio, device


@torch.no_grad()
def run_vae_on_video(
    input_video_path,
    output_dir=DEFAULT_OUTPUT_DIR,
    vae_path=DEFAULT_VAE_PATH,
    vae_precision="fp16",
    device=None,
    sample_posterior=False,
    enable_tiling=True,
    vae=None,
    resolved_vae_path=None,
    s_ratio=None,
    t_ratio=None,
    save_reconstruction=True,
):
    start_time = time.perf_counter()
    if vae is None:
        vae, resolved_vae_path, s_ratio, t_ratio, device = initialize_vae(
            vae_path=vae_path,
            vae_precision=vae_precision,
            device=device,
            enable_tiling=enable_tiling,
        )
    else:
        device = device or getattr(vae, "device", None) or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if resolved_vae_path is None or s_ratio is None or t_ratio is None:
            raise ValueError(
                "resolved_vae_path, s_ratio, and t_ratio must be provided when reusing a preloaded VAE."
            )

    video, fps = load_video_tensor(input_video_path)
    prepared_video, prep_metadata = prepare_video_for_vae(video)
    prepared_video = prepared_video.to(device=device, dtype=vae.dtype)

    posterior = vae.encode(prepared_video).latent_dist
    latent = posterior.sample() if sample_posterior else posterior.mode()
    reconstructed = None
    if save_reconstruction:
        reconstructed = vae.decode(latent).sample

    sample_name = Path(input_video_path).stem
    sample_output_dir = safe_dir(Path(output_dir) / sample_name)

    latent_path = sample_output_dir / f"{sample_name}_latent.pt"
    reconstructed_path = sample_output_dir / f"{sample_name}_reconstructed.mp4"
    prepared_input_path = sample_output_dir / f"{sample_name}_prepared_input.mp4"
    metadata_path = sample_output_dir / f"{sample_name}_metadata.json"

    latent_payload = {
        "latent": latent.detach().cpu(),
        "posterior_mean": posterior.mean.detach().cpu(),
        "posterior_logvar": posterior.logvar.detach().cpu(),
        "input_video_path": str(input_video_path),
        "vae_path": str(resolved_vae_path),
        "sample_posterior": sample_posterior,
        "fps": fps,
        "spatial_compression_ratio": s_ratio,
        "time_compression_ratio": t_ratio,
        "prepared_video_shape": list(prepared_video.shape),
    }
    torch.save(latent_payload, latent_path)

    if save_reconstruction:
        save_videos_grid(prepared_video.detach().cpu(), str(prepared_input_path), rescale=True, fps=fps)
        save_videos_grid(reconstructed.detach().cpu(), str(reconstructed_path), rescale=True, fps=fps)

    metadata = {
        "input_video_path": str(input_video_path),
        "vae_path": str(resolved_vae_path),
        "device": str(device),
        "vae_precision": vae_precision,
        "sample_posterior": sample_posterior,
        "fps": fps,
        "latent_path": str(latent_path),
        "prepared_input_path": str(prepared_input_path) if save_reconstruction else None,
        "reconstructed_path": str(reconstructed_path) if save_reconstruction else None,
        "prepared_video_shape": list(prepared_video.shape),
        "latent_shape": list(latent.shape),
        "posterior_mean_shape": list(posterior.mean.shape),
        "posterior_logvar_shape": list(posterior.logvar.shape),
        "reconstructed_shape": list(reconstructed.shape) if reconstructed is not None else None,
        "spatial_compression_ratio": s_ratio,
        "time_compression_ratio": t_ratio,
        "saved_reconstruction": save_reconstruction,
        "elapsed_seconds": time.perf_counter() - start_time,
        **prep_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return metadata


def process_video_range(
    start_index=DEFAULT_START_INDEX,
    end_index=DEFAULT_END_INDEX,
    output_dir=DEFAULT_OUTPUT_DIR,
    vae_path=DEFAULT_VAE_PATH,
    vae_precision="fp16",
    device=None,
    sample_posterior=False,
    enable_tiling=True,
    skip_missing=True,
    save_reconstruction=True,
):
    batch_start_time = time.perf_counter()
    vae, resolved_vae_path, s_ratio, t_ratio, resolved_device = initialize_vae(
        vae_path=vae_path,
        vae_precision=vae_precision,
        device=device,
        enable_tiling=enable_tiling,
    )
    all_metadata = []
    for video_index in range(start_index, end_index + 1):
        input_video_path = Path(video_index_to_path(video_index))
        if not input_video_path.exists():
            message = f"Skipping missing video: {input_video_path}"
            if skip_missing:
                print(message)
                continue
            raise FileNotFoundError(message)

        print(f"Processing {input_video_path} ...")
        metadata = run_vae_on_video(
            input_video_path=str(input_video_path),
            output_dir=output_dir,
            vae_path=vae_path,
            vae_precision=vae_precision,
            device=resolved_device,
            sample_posterior=sample_posterior,
            enable_tiling=enable_tiling,
            vae=vae,
            resolved_vae_path=resolved_vae_path,
            s_ratio=s_ratio,
            t_ratio=t_ratio,
            save_reconstruction=save_reconstruction,
        )
        all_metadata.append(metadata)
        print(
            f"Finished {input_video_path.name} in {metadata['elapsed_seconds']:.2f}s"
        )

    summary_path = Path(output_dir) / f"batch_summary_{start_index:04d}_{end_index:04d}.json"
    total_elapsed_seconds = time.perf_counter() - batch_start_time
    processed_count = len(all_metadata)
    summary = {
        "start_index": start_index,
        "end_index": end_index,
        "processed_count": processed_count,
        "total_elapsed_seconds": total_elapsed_seconds,
        "average_elapsed_seconds": (
            total_elapsed_seconds / processed_count if processed_count else 0.0
        ),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(
        json.dumps({"summary": summary, "videos": all_metadata}, indent=2)
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run only the HunyuanVideo VAE on an input video.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the input video.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save latents and reconstructions.")
    parser.add_argument("--vae-path", default=DEFAULT_VAE_PATH, help="Path to the VAE checkpoint directory.")
    parser.add_argument("--vae-precision", default="fp16", choices=["fp16", "bf16", "fp32"], help="Precision used to load the VAE.")
    parser.add_argument("--device", default=None, help="Device to run on, for example cuda or cpu.")
    parser.add_argument("--sample-posterior", action="store_true", help="Sample from the posterior instead of using the posterior mode.")
    parser.add_argument("--disable-tiling", action="store_true", help="Disable VAE tiling.")
    parser.add_argument("--process-all", action="store_true", help="Process the full indexed video range instead of a single input video.")
    parser.add_argument("--start-index", type=int, default=DEFAULT_START_INDEX, help="Start index for batch video processing.")
    parser.add_argument("--end-index", type=int, default=DEFAULT_END_INDEX, help="End index for batch video processing.")
    parser.add_argument("--strict-missing", action="store_true", help="Fail instead of skipping when a video in the batch range is missing.")
    parser.add_argument("--latent-only", action="store_true", help="Only save latent and metadata, skipping VAE decode and reconstructed video outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.process_all:
        summary = process_video_range(
            start_index=args.start_index,
            end_index=args.end_index,
            output_dir=args.output_dir,
            vae_path=args.vae_path,
            vae_precision=args.vae_precision,
            device=args.device,
            sample_posterior=args.sample_posterior,
            enable_tiling=not args.disable_tiling,
            skip_missing=not args.strict_missing,
            save_reconstruction=not args.latent_only,
        )
        print(json.dumps(summary, indent=2))
    else:
        metadata = run_vae_on_video(
            input_video_path=args.input,
            output_dir=args.output_dir,
            vae_path=args.vae_path,
            vae_precision=args.vae_precision,
            device=args.device,
            sample_posterior=args.sample_posterior,
            enable_tiling=not args.disable_tiling,
            save_reconstruction=not args.latent_only,
        )
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

    
