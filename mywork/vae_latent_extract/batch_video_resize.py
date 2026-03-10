import os
import subprocess
import sys

input_dir = "/projects/prjs1914/input/rescaled_final"
output_dir = "/projects/prjs1914/input/rescaled_final_540p"

os.makedirs(output_dir, exist_ok=True)
failed_videos = []

for idx in range(1, 2180):
    video_name = f"{idx:04d}_fw.mp4"
    input_path = os.path.join(input_dir, video_name)
    if not os.path.exists(input_path):
        print(f"[Warning] {input_path} not found, skipping.")
        continue
    cmd = [
        sys.executable,
        "/home/szhang2/AlchemyRepos/HunyuanVideo/mywork/vae_latent_extract/video_resize.py",
        "--input", input_path,
        "--output-dir", output_dir,
    ]
    print(f"Processing {input_path} ...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        failed_videos.append((input_path, exc.returncode))
        print(f"[Error] Failed to process {input_path} (exit code: {exc.returncode}).")

print("Batch resize finished.")
if failed_videos:
    print(f"[Summary] {len(failed_videos)} videos failed:")
    for path, returncode in failed_videos:
        print(f"  - {path} (exit code: {returncode})")
else:
    print("[Summary] All videos processed successfully.")
