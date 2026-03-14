from pathlib import Path

import torch


example_input_file = "/projects/prjs1914/output/vae_latent_540p/0001_fw_16by9_960x544_crop_540p/0001_fw_16by9_960x544_crop_540p_latent.pt"
output_dir = "/projects/prjs1914/output/vae_latent_540p_time_averaged"
input_root_dir = "/projects/prjs1914/output/vae_latent_540p"

Path(output_dir).mkdir(parents=True, exist_ok=True)


def _average_latent_tensor_over_time(tensor):
    if tensor is None:
        return None
    if tensor.ndim != 5:
        raise ValueError(f"Expected a 5D latent tensor shaped [B, C, T, H, W], got {list(tensor.shape)}")
    averaged = tensor.mean(dim=2)
    if averaged.shape[0] == 1:
        averaged = averaged.squeeze(0)
    return averaged

def average_latents_over_time(latent_file=example_input_file):
    """from   "latent_shape": [
    1,
    16,
    15,(the time dimension)
    68,
    120
  ],
    to   "latent_shape": [
    16,
    68,
    120
    ], by averaging over the time dimension.
        """
    latent_path = Path(latent_file)
    payload = torch.load(latent_path, map_location="cpu")

    averaged_latent = _average_latent_tensor_over_time(payload.get("latent"))
    averaged_posterior_mean = _average_latent_tensor_over_time(payload.get("posterior_mean"))
    averaged_posterior_logvar = _average_latent_tensor_over_time(payload.get("posterior_logvar"))

    output_payload = {
        "latent": averaged_latent,
        "posterior_mean": averaged_posterior_mean,
        "posterior_logvar": averaged_posterior_logvar,
        "input_latent_path": str(latent_path),
        "input_video_path": payload.get("input_video_path"),
        "vae_path": payload.get("vae_path"),
        "sample_posterior": payload.get("sample_posterior"),
        "fps": payload.get("fps"),
        "spatial_compression_ratio": payload.get("spatial_compression_ratio"),
        "time_compression_ratio": payload.get("time_compression_ratio"),
        "prepared_video_shape": payload.get("prepared_video_shape"),
        "original_latent_shape": list(payload["latent"].shape),
        "averaged_latent_shape": list(averaged_latent.shape),
        "reduction": "mean_over_time_dim_then_squeeze_batch_if_one",
    }

    output_path = Path(output_dir) / latent_path.parent.name
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / latent_path.name
    torch.save(output_payload, save_path)
    return save_path


def _index_to_latent_path(video_index):
    sample_name = f"{video_index:04d}_fw_16by9_960x544_crop_540p"
    return Path(input_root_dir) / sample_name / f"{sample_name}_latent.pt"


def main():
    processed = 0
    skipped = 0
    for video_index in range(1, 2180):
        latent_path = _index_to_latent_path(video_index)
        if not latent_path.exists():
            print(f"[Skip] Missing latent file: {latent_path}")
            skipped += 1
            continue

        save_path = average_latents_over_time(latent_path)
        processed += 1
        print(f"[Saved] {save_path}")

    print(f"Finished. Processed: {processed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
