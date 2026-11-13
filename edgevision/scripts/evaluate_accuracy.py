"""Evaluate detection count accuracy against human ground truth.

Computes Mean Absolute Error (MAE) per target class before and after
temporal count smoothing. If ground truth is not provided or unpopulated,
honestly reports 'NOT MEASURED'.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from edgevision.smoothing import apply_rolling_median_counts

TARGET_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}


def load_ground_truth(gt_csv_path: str | Path) -> Optional[Dict[int, Dict[str, int]]]:
    """Load ground truth annotations if available and populated.

    Returns:
        Dict mapping frame_idx -> {class_name: count}, or None if unpopulated/missing.
    """
    p = Path(gt_csv_path)
    if not p.is_file():
        return None

    gt_data: Dict[int, Dict[str, int]] = {}
    populated_count = 0

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_idx = int(row["frame"])
                # Check whether row has populated ground truth numbers
                has_values = any(
                    cls_name in row and row[cls_name].strip() != ""
                    for cls_name in TARGET_CLASSES
                )
                if not has_values:
                    continue

                counts: Dict[str, int] = {}
                for cls_name in TARGET_CLASSES:
                    val_str = row.get(cls_name, "").strip()
                    counts[cls_name] = int(val_str) if val_str else 0

                gt_data[frame_idx] = counts
                populated_count += 1
            except (ValueError, KeyError):
                continue

    if populated_count == 0:
        return None

    return gt_data


def evaluate_accuracy(
    pred_csv_path: str | Path,
    gt_csv_path: str | Path,
    smooth_window: int = 5,
    output_json: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """Compare predictions against ground truth and compute MAE metrics."""
    gt_map = load_ground_truth(gt_csv_path)

    print("=" * 80)
    print("EDGEVISION GROUND-TRUTH ACCURACY EVALUATION")
    print("=" * 80)

    if gt_map is None:
        print(f"Ground Truth File: '{gt_csv_path}'")
        print("Accuracy Status:   NOT MEASURED")
        print("Reason:            No ground truth counts found (file missing or unpopulated template).")
        print("To measure:        Populate 'ground_truth/ground_truth.csv' with integer counts.")
        print("=" * 80)
        return None

    pred_p = Path(pred_csv_path)
    if not pred_p.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {pred_csv_path}")

    # Read predictions
    pred_rows: List[Dict[str, Any]] = []
    with open(pred_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {"frame": int(row["frame"])}
            for cls_name in TARGET_CLASSES:
                parsed[cls_name] = int(row.get(cls_name, 0) or 0)
            parsed["total_vehicles"] = int(row.get("total_vehicles", 0) or sum(parsed[c] for c in VEHICLE_CLASSES))
            pred_rows.append(parsed)

    # Sort by frame index
    pred_rows.sort(key=lambda x: x["frame"])

    # Apply temporal smoothing
    smoothed_rows = apply_rolling_median_counts(pred_rows, window_size=smooth_window)

    pred_lookup = {r["frame"]: r for r in pred_rows}
    smooth_lookup = {r["frame"]: r for r in smoothed_rows}

    matched_frames = sorted([f for f in gt_map.keys() if f in pred_lookup])

    if not matched_frames:
        print("Error: No overlapping frame indices between ground truth and predictions.")
        return None

    mae_raw: Dict[str, float] = {}
    mae_smooth: Dict[str, float] = {}

    for cls_name in TARGET_CLASSES:
        diffs_raw = [abs(pred_lookup[f][cls_name] - gt_map[f][cls_name]) for f in matched_frames]
        diffs_smooth = [abs(smooth_lookup[f][cls_name] - gt_map[f][cls_name]) for f in matched_frames]
        mae_raw[cls_name] = float(np.mean(diffs_raw))
        mae_smooth[cls_name] = float(np.mean(diffs_smooth))

    # Vehicle total MAE
    v_diffs_raw = [
        abs(pred_lookup[f]["total_vehicles"] - sum(gt_map[f][c] for c in VEHICLE_CLASSES))
        for f in matched_frames
    ]
    v_diffs_smooth = [
        abs(smooth_lookup[f]["total_vehicles"] - sum(gt_map[f][c] for c in VEHICLE_CLASSES))
        for f in matched_frames
    ]
    mae_raw["total_vehicles"] = float(np.mean(v_diffs_raw))
    mae_smooth["total_vehicles"] = float(np.mean(v_diffs_smooth))

    # Format table
    print(f"Evaluated on {len(matched_frames)} annotated frames (smoothing window={smooth_window}):")
    print(f"{'Class':<20} | {'MAE (Raw)':<15} | {'MAE (Smoothed)':<15} | {'Improvement':<12}")
    print("-" * 70)
    for k in ["total_vehicles", "person", "car", "truck", "bus", "motorcycle", "bicycle"]:
        raw_val = mae_raw[k]
        sm_val = mae_smooth[k]
        diff = raw_val - sm_val
        delta_str = f"{diff:+.2f}" if diff != 0 else "0.00"
        print(f"{k:<20} | {raw_val:<15.3f} | {sm_val:<15.3f} | {delta_str:<12}")
    print("=" * 80)

    results = {
        "num_frames": len(matched_frames),
        "smooth_window": smooth_window,
        "mae_raw": mae_raw,
        "mae_smoothed": mae_smooth,
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection count accuracy vs ground truth.")
    parser.add_argument("--pred-csv", type=str, default="outputs/traffic_sample/counts.csv", help="Prediction CSV.")
    parser.add_argument("--gt-csv", type=str, default="ground_truth/ground_truth.csv", help="Ground truth CSV.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Smoothing window.")
    parser.add_argument("--out-json", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    evaluate_accuracy(
        pred_csv_path=args.pred_csv,
        gt_csv_path=args.gt_csv,
        smooth_window=args.smooth_window,
        output_json=args.out_json,
    )


if __name__ == "__main__":
    main()
