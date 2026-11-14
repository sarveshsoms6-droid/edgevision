"""Script to train or fine-tune YOLOv11n on custom vehicle detection datasets.

Supports datasets formatted in YOLO format (e.g., Alkan Erturan Vehicle Detection Dataset)
with classes: Ambulance, Bus, Car, Motorcycle, Truck.

Usage:
    python scripts/train_yolo11_custom.py --data dataset.yaml --epochs 100 --imgsz 640 --batch 16
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or fine-tune YOLOv11 on custom vehicle detection dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset.yaml",
        help="Path to dataset YAML configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n.pt",
        help="Base pretrained YOLO model weights (.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (e.g. 640)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Optimizer (AdamW, SGD, auto)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="CUDA device (e.g. 0) or 'cpu'",
    )
    parser.add_argument(
        "--output-weights",
        type=str,
        default="models/yolo11n_vehicle_custom.pt",
        help="Target destination for best trained weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print("[Error] Ultralytics is required for training: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.is_file():
        alt_path = _project_root / args.model
        if alt_path.is_file():
            model_path = alt_path
        else:
            model_path = Path("yolo11n.pt")

    data_path = Path(args.data)
    if not data_path.is_file():
        alt_data = _project_root / args.data
        if alt_data.is_file():
            data_path = alt_data

    print("=" * 70)
    print("YOLOv11 VEHICLE DETECTION TRAINING / FINE-TUNING")
    print("=" * 70)
    print(f"Base Model Weights: {model_path}")
    print(f"Dataset Config    : {data_path}")
    print(f"Target Epochs     : {args.epochs}")
    print(f"Batch Size        : {args.batch}")
    print(f"Resolution (imgsz): {args.imgsz}x{args.imgsz}")
    print(f"Optimizer         : {args.optimizer}")
    print("=" * 70)

    if not data_path.is_file():
        print(f"[Notice] Dataset file '{args.data}' not found.", file=sys.stderr)
        print("To download the Alkan Erturan Vehicle Detection dataset from Kaggle:", file=sys.stderr)
        print("  pip install kagglehub", file=sys.stderr)
        print("  python -c \"import kagglehub; path = kagglehub.dataset_download('alkanerturan/vehicledetection'); print(path)\"", file=sys.stderr)
        print("Then point --data to dataset.yaml inside the downloaded directory.")
        sys.exit(1)

    # Initialize model
    model = YOLO(str(model_path))

    # Run training
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": args.optimizer,
        "project": "runs/detect",
        "name": "train_vehicle",
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    print("[EdgeVision] Starting YOLOv11 training...")
    results = model.train(**train_kwargs)

    # Resolve best weights
    best_weights = Path("runs/detect/train_vehicle/weights/best.pt")
    if best_weights.is_file():
        out_dest = Path(args.output_weights)
        out_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(best_weights), str(out_dest))
        print(f"\n[Success] Training complete! Best weights saved to: {out_dest}")
    else:
        print("\n[Warning] Best weights file not located in default runs directory.")


if __name__ == "__main__":
    main()
