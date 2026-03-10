import argparse
import json
from pathlib import Path

import torch


DEFAULT_LATENT_FILE = "/projects/prjs1914/output/vae_latent/0001_fw/latent.pt"


def load_saved_latent(latent_file=DEFAULT_LATENT_FILE, map_location="cpu"):
	latent_file = Path(latent_file)
	payload = torch.load(latent_file, map_location=map_location)
	if not isinstance(payload, dict):
		raise TypeError(f"Expected a dict payload in {latent_file}, got {type(payload)}")
	return payload


def summarize_latent_payload(payload):
	summary = {}
	for key, value in payload.items():
		if isinstance(value, torch.Tensor):
			summary[key] = {
				"shape": list(value.shape),
				"dtype": str(value.dtype),
				"device": str(value.device),
				"min": float(value.min().item()),
				"max": float(value.max().item()),
				"mean": float(value.float().mean().item()),
				"std": float(value.float().std().item()),
			}
		else:
			summary[key] = value
	return summary


def parse_args():
	parser = argparse.ArgumentParser(description="Read a saved VAE latent payload and print a summary.")
	parser.add_argument("--latent-file", default=DEFAULT_LATENT_FILE, help="Path to latent.pt")
	parser.add_argument("--map-location", default="cpu", help="torch.load map_location")
	return parser.parse_args()


def main():
	args = parse_args()
	payload = load_saved_latent(args.latent_file, map_location=args.map_location)
	summary = summarize_latent_payload(payload)
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()