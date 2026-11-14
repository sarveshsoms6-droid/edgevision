"""Run YOLOv11 inference on all user input files stored in the testing/ folder.

Usage:
    python scripts/test_from_folder.py [--conf 0.40] [--output outputs/testing_results]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import cv2

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from edgevision.detector_yolo import YOLOv11Detector
from edgevision.annotate import annotate_frame
from edgevision.counter import count_objects_in_frame, compute_vehicle_total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv11 detection on all inputs stored in the testing/ folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="testing",
        help="Path to testing input folder",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n.pt",
        help="YOLO model weights path (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence cutoff threshold",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/testing_results",
        help="Output directory for annotated test outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_path = Path(args.test_dir)
    if not test_path.is_dir():
        test_path = _project_root / args.test_dir

    if not test_path.is_dir():
        print(f"[Error] Testing folder not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    # Collect test inputs
    input_files = [f for f in test_path.iterdir() if f.is_file()]

    print("=" * 70)
    print("YOLOv11 USER TESTING FOLDER EVALUATION")
    print("=" * 70)
    print(f"Testing Directory : {test_path.resolve()}")
    print(f"Total User Inputs : {len(input_files)}")
    print("=" * 70)

    if not input_files:
        print("[Notice] The testing folder is currently empty.")
        print("Every time you upload a video or image via the web app (http://127.0.0.1:5000)")
        print("or run 'python run.py --input <path>', it is automatically stored here.")
        print("=" * 70)
        return

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = _project_root / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOv11Detector(model_path=args.model, conf_threshold=args.conf)
    print(f"Model loaded: {args.model} ({detector.runtime_name})")

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    for item in input_files:
        ext = item.suffix.lower()
        print(f"\nEvaluating Input: {item.name}")

        if ext in image_exts:
            img = cv2.imread(str(item))
            if img is None:
                print(f"  [Warning] Could not decode image: {item.name}")
                continue

            boxes, scores, labels = detector.predict_detections(img, conf_threshold=args.conf)
            counts = count_objects_in_frame(labels, detector.id_to_class_map)
            total_v = compute_vehicle_total(counts)

            counts_repr = ", ".join(f"{k}: {v}" for k, v in counts.items() if v > 0) or "None"
            print(f"  Detections: Vehicles={total_v}, Pedestrians={counts.get('person', 0)} ({counts_repr})")

            annotated = annotate_frame(
                frame=img,
                boxes=boxes,
                scores=scores,
                class_ids=labels,
                class_name_map=detector.id_to_class_map,
                counts=counts,
            )
            out_img = out_dir / f"annotated_{item.name}"
            cv2.imwrite(str(out_img), annotated)
            print(f"  Saved annotation -> {out_img}")

        elif ext in video_exts:
            from edgevision.config import load_config
            from edgevision.pipeline import EdgeVisionPipeline

            cfg = load_config(_project_root / "config.yaml")
            cfg.model.runtime = "yolov11"
            cfg.conf_threshold = args.conf
            pipeline = EdgeVisionPipeline(cfg)
            vid_out = out_dir / item.stem
            res = pipeline.process_video(item, vid_out, runtime_override="yolov11")
            print(f"  Video Processed: {res.processed_frames} frames in {res.processing_time_seconds}s ({res.end_to_end_fps} FPS)")
            print(f"  Peak Vehicles: {res.peak_counts.get('vehicles', 0)}, Peak Persons: {res.peak_counts.get('person', 0)}")
            print(f"  Saved artifacts -> {vid_out}")


if __name__ == "__main__":
    main()
