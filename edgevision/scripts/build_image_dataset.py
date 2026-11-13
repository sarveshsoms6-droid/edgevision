"""Build an enriched traffic image dataset from uploaded and sample videos.

Extracts diverse, stratified frames across all video clips in uploads/ and sample_data/
to create an image dataset for model training, fine-tuning, and INT8 calibration.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    num_frames: int = 25,
    max_width: int = 640,
    prefix: str = "",
) -> int:
    """Extract evenly spaced diverse frames from a single video file."""
    if not video_path.is_file():
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0

    actual_extract = min(num_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, actual_extract, dtype=int)
    extracted_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    current_idx = 0

    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        if w > max_width:
            new_h = int(h * (max_width / w))
            frame = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_LINEAR)

        # Skip near-black or corrupted frames
        if np.mean(frame) < 8.0:
            continue

        filename = f"{prefix}_{target_idx:05d}.jpg" if prefix else f"frame_{current_idx:05d}.jpg"
        cv2.imwrite(str(output_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted_count += 1
        current_idx += 1

    cap.release()
    return extracted_count


def build_image_dataset(
    uploads_dir: str = "uploads",
    sample_dir: str = "sample_data",
    output_dir: str = "dataset/images",
    frames_per_video: int = 20,
    max_total_images: int = 600,
) -> int:
    """Extract diverse frames from all videos into a centralized image dataset."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    video_paths = []
    up_path = Path(uploads_dir)
    if up_path.is_dir():
        video_paths.extend(sorted(up_path.glob("*.mp4")))
    samp_path = Path(sample_dir)
    if samp_path.is_dir():
        video_paths.extend(sorted(samp_path.glob("*.mp4")))

    print("=" * 70)
    print("BUILDING DIVERSE TRAFFIC IMAGE DATASET")
    print(f"Source Videos Found   : {len(video_paths)}")
    print(f"Target Destination    : {dest}")
    print(f"Frames per Video Target: {frames_per_video}")
    print(f"Maximum Total Images  : {max_total_images}")
    print("=" * 70)

    total_extracted = 0
    for idx, v_path in enumerate(video_paths):
        if total_extracted >= max_total_images:
            break
        prefix = f"vid{idx:02d}_{v_path.stem[:8]}"
        needed = min(frames_per_video, max_total_images - total_extracted)
        count = extract_frames_from_video(
            video_path=v_path,
            output_dir=dest,
            num_frames=needed,
            prefix=prefix,
        )
        total_extracted += count
        print(f"[{idx+1}/{len(video_paths)}] {v_path.name}: extracted {count} frames (Total: {total_extracted})")

    print("=" * 70)
    print(f"DATASET EXTRACTION COMPLETE: {total_extracted} real traffic images saved in {dest}")
    print("=" * 70)
    return total_extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract diverse frames into image dataset.")
    parser.add_argument("--uploads", default="uploads", help="Path to uploads directory")
    parser.add_argument("--output", default="dataset/images", help="Output image directory")
    parser.add_argument("--frames-per-video", type=int, default=20, help="Frames per video")
    parser.add_argument("--max-total", type=int, default=600, help="Max total images to extract")
    args = parser.parse_args()

    build_image_dataset(
        uploads_dir=args.uploads,
        output_dir=args.output,
        frames_per_video=args.frames_per_video,
        max_total_images=args.max_total,
    )
