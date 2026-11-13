"""EdgeVision CLI Entry Point.

Usage:
    python run.py --input clip.mp4 --runtime pytorch-reference
    python run.py --input clip.mp4 --runtime executorch-fp32
    python run.py --input clip.mp4 --runtime executorch-int8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

from edgevision.config import load_config
from edgevision.pipeline import EdgeVisionPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EdgeVision: Offline Traffic Video Analysis for Raspberry Pi 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video file (mp4, avi, mov, mkv)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["executorch-int8", "executorch-fp32", "pytorch-reference"],
        default=None,
        help="Override inference runtime specified in config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (defaults to outputs/<job_id>)",
    )
    parser.add_argument(
        "--stride",
        type=str,
        default=None,
        help="Frame stride: integer >= 1 or 'auto'",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of XNNPACK execution threads (1-4)",
    )
    parser.add_argument(
        "--zone",
        action="store_true",
        help="Enable polygon Region of Interest (ROI) zone counting",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI execution flow."""
    args = parse_args()

    print(f"[EdgeVision] Loading configuration: {args.config}")
    config = load_config(args.config)

    if args.runtime:
        config.model.runtime = args.runtime
    if args.stride:
        if args.stride.lower() == "auto":
            config.frame_stride = "auto"
        else:
            config.frame_stride = int(args.stride)
    if args.threads:
        config.threads = args.threads
    if args.zone:
        config.zone.enabled = True

    config.validate()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[EdgeVision] Error: Input video file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    job_id = uuid.uuid4().hex[:8]
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"job_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EDGEVISION OFFLINE DETECTION PIPELINE")
    print(f"Target Video : {input_path}")
    print(f"Runtime      : {config.model.runtime}")
    print(f"Frame Stride : {config.frame_stride}")
    print(f"Output Dir   : {output_dir}")
    print("=" * 70)

    pipeline = EdgeVisionPipeline(config)
    result = pipeline.process_video(
        video_path=input_path,
        output_dir=output_dir,
        runtime_override=config.model.runtime,
    )

    print("=" * 70)
    print("PROCESSING COMPLETED SUCCESSFULLY")
    print(f"Runtime Used       : {result.runtime_used}")
    print(f"Total Frames       : {result.total_frames}")
    print(f"Processed Frames   : {result.processed_frames} | Skipped: {result.skipped_frames}")
    print(f"Total Time         : {result.processing_time_seconds:.2f} s")
    print(f"End-to-End Speed   : {result.end_to_end_fps:.1f} FPS (Effective: {result.effective_fps:.1f} FPS)")
    print(f"Peak Vehicle Count : {result.peak_counts.get('vehicles', 0)}")
    print(f"Peak Person Count  : {result.peak_counts.get('person', 0)}")
    print(f"CSV Report Saved   : {result.csv_path}")
    print(f"Sample Frames Saved: {len(result.saved_sample_paths)}")
    for s_path in result.saved_sample_paths:
        print(f"  - {s_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
