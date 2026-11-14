"""Train or fine-tune YOLOv11 using images placed in the training/ folder.

Usage:
    python scripts/train_from_folder.py [--epochs 50] [--imgsz 640] [--batch 16]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import yaml

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 using images from the training/ folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="training",
        help="Path to training images folder",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n.pt",
        help="Pretrained YOLO base model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Error] Ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    train_path = Path(args.train_dir)
    if not train_path.is_dir():
        train_path = _project_root / args.train_dir

    if not train_path.is_dir():
        print(f"[Error] Training directory not found at: {train_path}", file=sys.stderr)
        sys.exit(1)

    # Search for .png and other images in training directory
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    images = [f for f in train_path.iterdir() if f.is_file() and f.suffix.lower() in image_exts]

    if not images:
        print("=" * 70)
        print("YOLOv11 CUSTOM TRAINING FOLDER STATUS")
        print("=" * 70)
        print(f"Directory: {train_path.resolve()}")
        print("Status   : STRICTLY EMPTY (No images found)")
        print("\nInstructions:")
        print("1. Paste your training .png images into this 'training' folder.")
        print("2. (Optional) Paste corresponding YOLO format annotation .txt files.")
        print("3. Run this script again: python scripts/train_from_folder.py")
        print("=" * 70)
        return

    print("=" * 70)
    print(f"Found {len(images)} training image(s) in: {train_path.resolve()}")
    print("Preparing YOLO training dataset configuration...")

    # Build dataset YAML
    data_yaml_path = _project_root / "training" / "dataset.yaml"
    dataset_dict = {
        "path": str(train_path.resolve()),
        "train": str(train_path.resolve()),
        "val": str(train_path.resolve()),
        "names": {
            0: "ambulance",
            1: "bus",
            2: "car",
            3: "motorcycle",
            4: "truck",
        },
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_dict, f, default_flow_style=False)

    model_file = Path(args.model)
    if not model_file.is_file():
        model_file = _project_root / args.model

    print(f"Base model : {model_file}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch}")
    print(f"Image size : {args.imgsz}")
    print("=" * 70)

    model = YOLO(str(model_file))
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project="runs/detect",
        name="train_custom_folder",
        exist_ok=True,
    )

    best_p = Path("runs/detect/train_custom_folder/weights/best.pt")
    if best_p.is_file():
        target_pt = _project_root / "models" / "yolo11n_vehicle_custom.pt"
        import shutil
        shutil.copy2(str(best_p), str(target_pt))
        print(f"\n[Success] Training completed! Model saved to: {target_pt}")


if __name__ == "__main__":
    main()
