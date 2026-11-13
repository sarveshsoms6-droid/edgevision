"""Script to extract high-resolution, uncompressed/high-quality frames from all uploaded traffic videos
to build a rich dataset of ~500 MB (1,500-2,500 high-resolution frames).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
import cv2
import numpy as np


def build_large_traffic_dataset(
    video_dir: str = "uploads",
    output_dir: str = "dataset/images",
    target_mb: float = 500.0,
    max_frames_per_video: int = 150,
) -> None:
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    print(f"Scanning {len(videos)} videos from '{video_dir}'...")

    total_saved = 0
    total_bytes = sum(f.stat().st_size for f in out_p.glob("*.jpg"))

    print(f"Initial dataset size: {total_bytes / (1024*1024):.2f} MB")

    for v_idx, v_path in enumerate(videos):
        if (total_bytes / (1024 * 1024)) >= target_mb:
            print(f"Reached target dataset size: {total_bytes / (1024*1024):.2f} MB!")
            break

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            continue

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            cap.release()
            continue

        # Extract frames spaced evenly
        count_to_extract = min(max_frames_per_video, n_frames)
        step = max(1, n_frames // count_to_extract)
        v_stem = Path(v_path).stem

        extracted_from_this = 0
        for f_idx in range(0, n_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Skip near-black frames
            if np.mean(frame) < 10.0:
                continue

            # Keep resolution sharp (up to 1280px wide) with high JPEG quality (95)
            h, w = frame.shape[:2]
            if w > 1280:
                new_h = int(h * (1280 / w))
                frame = cv2.resize(frame, (1280, new_h), interpolation=cv2.INTER_LINEAR)

            out_filename = out_p / f"{v_stem}_f{f_idx:05d}.jpg"
            if not out_filename.exists():
                cv2.imwrite(str(out_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                file_size = out_filename.stat().st_size
                total_bytes += file_size
                total_saved += 1
                extracted_from_this += 1

            if (total_bytes / (1024 * 1024)) >= target_mb:
                break

        cap.release()
        print(f"[{v_idx+1}/{len(videos)}] {Path(v_path).name}: extracted {extracted_from_this} frames. Running dataset total: {total_bytes / (1024*1024):.2f} MB")

    print("=" * 70)
    print(f"EXTRACTION COMPLETE!")
    print(f"Total images in dataset: {len(list(out_p.glob('*.jpg')))}")
    print(f"Total dataset disk size: {total_bytes / (1024*1024):.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    build_large_traffic_dataset(target_mb=500.0)
