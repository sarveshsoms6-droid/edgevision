"""Generate ground-truth annotation template and sample images.

Selects 10 to 15 evenly spaced frames from a video, exports them as high-quality
JPEG images, and generates a skeleton CSV template for human annotation.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import cv2
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def create_ground_truth_template(
    video_path: str | Path,
    output_dir: str | Path = "ground_truth",
    num_samples: int = 12,
) -> None:
    """Extract sample frames and generate empty ground truth CSV template.

    Args:
        video_path: Path to target input video.
        output_dir: Destination directory for samples and CSV template.
        num_samples: Number of evenly spaced frames (10 to 15).
    """
    v_path = Path(video_path)
    if not v_path.is_file():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    out_d = Path(output_dir)
    samples_dir = out_d / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video contains 0 frames: {video_path}")

    num_samples = max(2, min(num_samples, total_frames))
    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    rows = []
    print(f"Extracting {len(sample_indices)} sample frames from '{v_path.name}' ({total_frames} total frames)...")

    for f_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        timestamp_s = round(float(f_idx) / fps, 2)
        img_filename = f"frame_{f_idx:04d}.jpg"
        img_path = samples_dir / img_filename
        cv2.imwrite(str(img_path), frame)

        rows.append({
            "frame": int(f_idx),
            "timestamp_s": timestamp_s,
            "person": "",
            "bicycle": "",
            "car": "",
            "motorcycle": "",
            "bus": "",
            "truck": "",
        })

    cap.release()

    csv_path = out_d / "ground_truth_template.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "timestamp_s", "person", "bicycle", "car", "motorcycle", "bus", "truck"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("GROUND TRUTH TEMPLATE GENERATED")
    print("=" * 80)
    print(f"Sample images saved to:  {samples_dir}")
    print(f"Template CSV saved to:   {csv_path}")
    print("Instructions: Open the CSV in any spreadsheet editor or text editor,")
    print("inspect the corresponding images in 'ground_truth/samples/', fill in")
    print("the integer counts for each object class, and save the file.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ground-truth CSV template and sample frames.")
    parser.add_argument("--video", type=str, default="sample_data/traffic_sample.mp4", help="Input video path.")
    parser.add_argument("--out-dir", type=str, default="ground_truth", help="Destination folder.")
    parser.add_argument("--samples", type=int, default=12, help="Number of sample frames (10-15).")
    args = parser.parse_args()

    create_ground_truth_template(
        video_path=args.video,
        output_dir=args.out_d if hasattr(args, "out_d") else args.out_dir,
        num_samples=args.samples,
    )


if __name__ == "__main__":
    main()
