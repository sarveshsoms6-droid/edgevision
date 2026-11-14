"""Evaluate YOLOv11 model metrics (Precision, Recall, mAP50, mAP50-95).

Usage:
    python scripts/evaluate_yolo.py --weights models/yolo11n.pt --data dataset.yaml --imgsz 640
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv11 detection accuracy metrics (Precision, Recall, mAP50)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/yolo11n.pt",
        help="Path to YOLO weights (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset.yaml",
        help="Dataset YAML configuration file",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence cutoff threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Error] Ultralytics required: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        alt = _project_root / args.weights
        if alt.is_file():
            weights_path = alt

    if not weights_path.is_file():
        print(f"[Error] Weights file not found: {args.weights}", file=sys.stderr)
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.is_file():
        alt = _project_root / args.data
        if alt.is_file():
            data_path = alt

    if not data_path.is_file():
        print(f"[Notice] Dataset file '{args.data}' not found. Cannot compute validation set mAP.", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
    )

    print("=" * 70)
    print("YOLOv11 MODEL EVALUATION RESULTS")
    print("=" * 70)
    box_p = float(metrics.box.p.mean()) if hasattr(metrics.box, "p") else None
    box_r = float(metrics.box.r.mean()) if hasattr(metrics.box, "r") else None
    map50 = float(metrics.box.map50) if hasattr(metrics.box, "map50") else None
    map50_95 = float(metrics.box.map) if hasattr(metrics.box, "map") else None

    if box_p is not None:
        print(f"Mean Box Precision (P) : {box_p * 100:.2f}%")
    if box_r is not None:
        print(f"Mean Box Recall (R)    : {box_r * 100:.2f}%")
    if map50 is not None:
        print(f"mAP @ 0.50             : {map50 * 100:.2f}%")
    if map50_95 is not None:
        print(f"mAP @ 0.50:0.95        : {map50_95 * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
