"""Report generation and visualization module.

Generates:
- summary.md: peak counts, averages, vehicle-to-pedestrian ratio, congestion classification,
  and rule-based urban planning insights.
- Count-over-time trend chart saved as PNG using matplotlib (headless Agg backend).
- Failure-case exports when ground truth annotations are present.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure matplotlib uses headless Agg backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def determine_congestion_level(
    average_vehicles: float,
    low_max: float = 5.0,
    medium_max: float = 12.0,
) -> str:
    """Classify traffic congestion based on average vehicles detected per frame.

    Args:
        average_vehicles: Average count of vehicles per frame.
        low_max: Upper bound for Low congestion.
        medium_max: Upper bound for Medium congestion.

    Returns:
        One of 'Low', 'Medium', or 'High'.
    """
    if average_vehicles <= low_max:
        return "Low"
    elif average_vehicles <= medium_max:
        return "Medium"
    else:
        return "High"


def generate_planning_insight(
    vehicle_avg: float,
    pedestrian_avg: float,
    v_to_p_ratio: float,
    congestion_level: str,
) -> str:
    """Generate a single concise, rule-based urban planning insight sentence.

    Args:
        vehicle_avg: Average vehicle count per frame.
        pedestrian_avg: Average pedestrian count per frame.
        v_to_p_ratio: Ratio of vehicles to pedestrians.
        congestion_level: Evaluated congestion level string.

    Returns:
        A plain-language planning recommendation.
    """
    if congestion_level == "High":
        if pedestrian_avg >= 4.0:
            return (
                "Heavy vehicular density paired with sustained pedestrian volume warrants "
                "actuated crosswalks and traffic-calming geometry."
            )
        else:
            return (
                "High vehicle saturation indicates corridor bottlenecks; consider optimizing "
                "signal phase splits and arterial progression."
            )
    elif congestion_level == "Medium":
        if v_to_p_ratio > 3.0:
            return (
                "Moderate vehicular throughput with low pedestrian presence indicates current "
                "signal timings operate efficiently."
            )
        else:
            return (
                "Balanced multi-modal demand suggests protected pedestrian turn phases would "
                "improve safety without degrading vehicle throughput."
            )
    else:  # Low
        if pedestrian_avg > vehicle_avg and pedestrian_avg > 2.0:
            return (
                "Pedestrian volume exceeds vehicle traffic; prioritize sidewalk width and "
                "active-mobility infrastructure."
            )
        else:
            return (
                "Corridor operates with low vehicle and pedestrian volume, well within baseline capacity."
            )


def plot_counts_over_time(
    per_frame_counts: List[Dict[str, Any]],
    output_chart_path: str | Path,
) -> None:
    """Plot vehicle and pedestrian counts over time and save as PNG.

    Args:
        per_frame_counts: Chronological list of per-frame count records.
        output_chart_path: File destination for rendered PNG chart.
    """
    if not per_frame_counts:
        return

    out_p = Path(output_chart_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    timestamps = [
        float(r.get("timestamp_s", float(r.get("frame", i)) / 30.0))
        for i, r in enumerate(per_frame_counts)
    ]
    vehicles = [int(r.get("total_vehicles", 0)) for r in per_frame_counts]
    pedestrians = [int(r.get("person", 0)) for r in per_frame_counts]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    ax.plot(timestamps, vehicles, label="Vehicles (total)", color="#2ecc71", linewidth=2.0)
    ax.plot(timestamps, pedestrians, label="Pedestrians (person)", color="#3498db", linewidth=2.0)

    # Optional zone curves if present
    if "in_zone_vehicles" in per_frame_counts[0]:
        z_veh = [int(r.get("in_zone_vehicles", 0)) for r in per_frame_counts]
        ax.plot(timestamps, z_veh, label="In-Zone Vehicles", color="#e67e22", linestyle="--", linewidth=1.5)

    ax.set_title("Traffic Count Over Time", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Count per Frame", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    fig.savefig(str(out_p))
    plt.close(fig)


def generate_summary_markdown(
    results_dict: Dict[str, Any],
    output_path: str | Path,
) -> str:
    """Render and write the summary.md report.

    Args:
        results_dict: Pipeline results dictionary containing counts and telemetry.
        output_path: Path where summary.md will be saved.

    Returns:
        Generated markdown string.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    peak = results_dict.get("peak_counts", {})
    avg = results_dict.get("average_counts", {})
    congestion = results_dict.get("congestion_level", "Low")
    runtime = results_dict.get("runtime_used", "unknown")
    total_frames = results_dict.get("total_frames", 0)
    processed_frames = results_dict.get("processed_frames", 0)
    e2e_fps = results_dict.get("end_to_end_fps", 0.0)
    proc_time = results_dict.get("processing_time_seconds", 0.0)

    veh_avg = float(avg.get("vehicles", 0.0))
    ped_avg = float(avg.get("person", 0.0))
    v_to_p_ratio = round(veh_avg / max(ped_avg, 0.01), 2) if ped_avg > 0 else (round(veh_avg, 2) if veh_avg > 0 else 0.0)

    insight = generate_planning_insight(veh_avg, ped_avg, v_to_p_ratio, congestion)

    bench = results_dict.get("benchmark_metrics")
    latency_p50 = f"{bench.latency_p50_ms:.1f} ms" if bench else "NOT MEASURED"
    peak_rss = f"{bench.peak_rss_mb:.1f} MB" if bench else "NOT MEASURED"

    md_lines = [
        "# EdgeVision Traffic Analysis Summary",
        "",
        "## Executive Overview",
        f"- **Congestion Classification**: **{congestion}** (Average {veh_avg:.1f} vehicles/frame)",
        f"- **Vehicle-to-Pedestrian Ratio**: **{v_to_p_ratio:.2f}**",
        f"- **Planning Insight**: *\"{insight}\"*",
        "",
        "## Object Detection & Count Metrics",
        "",
        "| Class | Peak Count | Average Count (per frame) |",
        "|---|---|---|",
        f"| **Total Vehicles** | **{peak.get('vehicles', 0)}** | **{veh_avg:.2f}** |",
        f"| Person (Pedestrian) | {peak.get('person', 0)} | {ped_avg:.2f} |",
        f"| Car | {peak.get('car', 0)} | {avg.get('car', 0.0):.2f} |",
        f"| Truck | {peak.get('truck', 0)} | {avg.get('truck', 0.0):.2f} |",
        f"| Bus | {peak.get('bus', 0)} | {avg.get('bus', 0.0):.2f} |",
        f"| Motorcycle | {peak.get('motorcycle', 0)} | {avg.get('motorcycle', 0.0):.2f} |",
        f"| Bicycle | {peak.get('bicycle', 0)} | {avg.get('bicycle', 0.0):.2f} |",
        "",
        "## Performance & Telemetry",
        f"- **Inference Runtime**: `{runtime}`",
        f"- **Frames**: {processed_frames} processed / {total_frames} total",
        f"- **Total Time**: {proc_time:.2f} seconds",
        f"- **End-to-End Speed**: {e2e_fps:.1f} FPS",
        f"- **Inference Latency (p50)**: {latency_p50}",
        f"- **Peak RSS Memory**: {peak_rss}",
        "",
        "## Count Trends Over Time",
        "![Traffic Count Over Time](counts_chart.png)",
        "",
    ]

    content = "\n".join(md_lines)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def export_failure_cases(
    per_frame_counts: List[Dict[str, Any]],
    ground_truth_csv_path: str | Path,
    video_path: str | Path,
    output_dir: str | Path,
    top_k: int = 3,
) -> List[str]:
    """Identify and export frames with the highest count discrepancies vs ground truth.

    Args:
        per_frame_counts: Model predicted counts per frame.
        ground_truth_csv_path: Path to user-completed ground truth CSV.
        video_path: Source video file path.
        output_dir: Destination folder for exported failure frames.
        top_k: Number of highest-error sample frames to save.

    Returns:
        List of saved failure frame image paths.
    """
    gt_path = Path(ground_truth_csv_path)
    if not gt_path.is_file():
        return []

    # Read ground truth
    gt_map: Dict[int, Dict[str, int]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                f_idx = int(row["frame"])
                # Check if ground truth contains numbers or is empty template
                person_val = row.get("person", "").strip()
                if not person_val:
                    continue
                gt_map[f_idx] = {
                    "person": int(person_val),
                    "car": int(row.get("car", 0) or 0),
                    "truck": int(row.get("truck", 0) or 0),
                    "bus": int(row.get("bus", 0) or 0),
                    "motorcycle": int(row.get("motorcycle", 0) or 0),
                    "bicycle": int(row.get("bicycle", 0) or 0),
                }
            except (ValueError, KeyError):
                continue

    if not gt_map:
        return []

    pred_map = {int(r["frame"]): r for r in per_frame_counts if "frame" in r}

    errors: List[Tuple[float, int]] = []
    classes = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]

    for f_idx, gt_counts in gt_map.items():
        if f_idx in pred_map:
            p_row = pred_map[f_idx]
            err = sum(abs(int(p_row.get(cls, 0)) - gt_counts.get(cls, 0)) for cls in classes)
            errors.append((err, f_idx))

    errors.sort(key=lambda x: x[0], reverse=True)
    top_failures = errors[:top_k]
    if not top_failures or top_failures[0][0] == 0:
        return []

    fail_dir = Path(output_dir) / "failure_cases"
    fail_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    saved_paths: List[str] = []

    for err_val, f_idx in top_failures:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            out_file = fail_dir / f"failure_frame_{f_idx:04d}_err_{int(err_val)}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved_paths.append(str(out_file))

    cap.release()
    return saved_paths
