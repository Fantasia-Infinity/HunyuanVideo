from pathlib import Path

import torch


example_input_file = "/projects/prjs1914/output/vae_latent_540p/0001_fw_16by9_960x544_crop_540p/0001_fw_16by9_960x544_crop_540p_latent.pt"
output_dir = "/projects/prjs1914/output/vae_latent_540p_flattened"
input_root_dir = "/projects/prjs1914/output/vae_latent_540p"

Path(output_dir).mkdir(parents=True, exist_ok=True)


def _flatten_tensor(tensor):
    if tensor is None:
        return None, None
    contiguous_tensor = tensor.detach().cpu().contiguous()
    return contiguous_tensor.reshape(-1), list(contiguous_tensor.shape)


def _restore_tensor(flattened_tensor, original_shape):
    if flattened_tensor is None:
        return None
    if original_shape is None:
        raise ValueError("Missing original shape for flattened tensor reconstruction.")
    expected_numel = 1
    for dim in original_shape:
        expected_numel *= dim
    if flattened_tensor.numel() != expected_numel:
        raise ValueError(
            f"Flattened tensor has {flattened_tensor.numel()} elements, expected {expected_numel} for shape {original_shape}."
        )
    return flattened_tensor.reshape(original_shape)


def flatten_latent(latent_file=example_input_file):
    latent_path = Path(latent_file)
    payload = torch.load(latent_path, map_location="cpu")

    flattened_latent, latent_shape = _flatten_tensor(payload.get("latent"))
    flattened_posterior_mean, posterior_mean_shape = _flatten_tensor(payload.get("posterior_mean"))
    flattened_posterior_logvar, posterior_logvar_shape = _flatten_tensor(payload.get("posterior_logvar"))

    output_payload = {
        "latent": flattened_latent,
        "posterior_mean": flattened_posterior_mean,
        "posterior_logvar": flattened_posterior_logvar,
        "input_latent_path": str(latent_path),
        "input_video_path": payload.get("input_video_path"),
        "vae_path": payload.get("vae_path"),
        "sample_posterior": payload.get("sample_posterior"),
        "fps": payload.get("fps"),
        "spatial_compression_ratio": payload.get("spatial_compression_ratio"),
        "time_compression_ratio": payload.get("time_compression_ratio"),
        "prepared_video_shape": payload.get("prepared_video_shape"),
        "original_shapes": {
            "latent": latent_shape,
            "posterior_mean": posterior_mean_shape,
            "posterior_logvar": posterior_logvar_shape,
        },
        "flatten_method": "contiguous_row_major",
    }

    output_path = Path(output_dir) / latent_path.parent.name
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / latent_path.name.replace("_latent.pt", "_latent_flattened.pt")
    torch.save(output_payload, save_path)
    return save_path


def reconstruct_latent_from_flattened(flattened_latent_file):
    flattened_path = Path(flattened_latent_file)
    payload = torch.load(flattened_path, map_location="cpu")

    original_shapes = payload.get("original_shapes")
    if original_shapes is None:
        raise ValueError(f"Missing original_shapes in flattened latent file: {flattened_path}")

    restored_payload = {
        "latent": _restore_tensor(payload.get("latent"), original_shapes.get("latent")),
        "posterior_mean": _restore_tensor(
            payload.get("posterior_mean"), original_shapes.get("posterior_mean")
        ),
        "posterior_logvar": _restore_tensor(
            payload.get("posterior_logvar"), original_shapes.get("posterior_logvar")
        ),
        "input_latent_path": payload.get("input_latent_path"),
        "input_video_path": payload.get("input_video_path"),
        "vae_path": payload.get("vae_path"),
        "sample_posterior": payload.get("sample_posterior"),
        "fps": payload.get("fps"),
        "spatial_compression_ratio": payload.get("spatial_compression_ratio"),
        "time_compression_ratio": payload.get("time_compression_ratio"),
        "prepared_video_shape": payload.get("prepared_video_shape"),
        "restored_from_flattened_path": str(flattened_path),
        "restored_shapes": original_shapes,
    }

    save_path = flattened_path.with_name(
        flattened_path.name.replace("_latent_flattened.pt", "_latent_restored.pt")
    )
    torch.save(restored_payload, save_path)
    return save_path


def _index_to_latent_path(video_index):
    sample_name = f"{video_index:04d}_fw_16by9_960x544_crop_540p"
    return Path(input_root_dir) / sample_name / f"{sample_name}_latent.pt"


def flatten_latents_for_range(start_index=1, end_index=2179):
    processed = 0
    skipped = 0
    saved_paths = []

    for video_index in range(start_index, end_index + 1):
        latent_path = _index_to_latent_path(video_index)
        if not latent_path.exists():
            print(f"[Skip] Missing latent file: {latent_path}")
            skipped += 1
            continue

        save_path = flatten_latent(latent_path)
        saved_paths.append(str(save_path))
        processed += 1
        print(f"[Saved] {save_path}")

    summary = {
        "start_index": start_index,
        "end_index": end_index,
        "processed": processed,
        "skipped": skipped,
        "saved_paths": saved_paths,
    }
    print(f"Finished. Processed: {processed}, skipped: {skipped}")
    return summary

