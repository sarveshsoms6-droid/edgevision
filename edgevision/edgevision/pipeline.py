"""End-to-end detection and counting pipeline coordinator with latency optimizations.

Orchestrates:
- Two-thread pipeline: bounded queue decoder thread + inference thread.
- Preprocessing and buffer management (320x320 cv2.INTER_LINEAR).
- Adaptive frame stride to maintain near-real-time throughput on edge hardware.
- Runtime inference (PyTorch reference, ExecuTorch FP32, or ExecuTorch INT8).
- Post-processing, thresholding, and vectorised NMS.
- Frame counting, temporal smoothing, and optional zone metrics.
- Performance statistics collection (p50, p95, p99, peak RSS, CPU temp).
- Artifact generation: CSV, annotated samples, benchmark.json, summary.md, counts_chart.png.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from edgevision.stream_utils import patch_stdout_stderr, safe_print
patch_stdout_stderr()

from edgevision.annotate import annotate_frame
from edgevision.benchmark import (
    BenchmarkMetrics,
    calculate_latency_percentiles,
    get_peak_rss_mb,
    print_benchmark_table,
    read_cpu_temperature,
)
from edgevision.config import EdgeVisionConfig
from edgevision.counter import FrameCounts, compute_vehicle_total, count_objects_in_frame
from edgevision.detector_pytorch import PyTorchDetector
from edgevision.postprocess import postprocess_detections
from edgevision.preprocess import Preprocessor
from edgevision.report import (
    determine_congestion_level,
    export_failure_cases,
    generate_summary_markdown,
    plot_counts_over_time,
)
from edgevision.smoothing import apply_rolling_median_counts
from edgevision.video_io import VideoDecoderThread, get_video_metadata, read_video_frames
from edgevision.zones import count_in_zone


@dataclass
class PipelineResult:
    """Aggregated output from an EdgeVision video processing run."""

    video_path: str
    runtime_used: str
    total_frames: int
    processed_frames: int
    skipped_frames: int
    fps_video: float
    effective_fps: float
    end_to_end_fps: float
    processing_time_seconds: float
    per_frame_counts: List[Dict[str, Any]] = field(default_factory=list)
    peak_counts: Dict[str, int] = field(default_factory=dict)
    average_counts: Dict[str, float] = field(default_factory=dict)
    congestion_level: str = "Low"
    benchmark_metrics: Optional[BenchmarkMetrics] = None
    saved_sample_paths: List[str] = field(default_factory=list)
    output_dir: str = ""
    csv_path: str = ""
    benchmark_json_path: str = ""
    summary_md_path: str = ""
    chart_path: str = ""
    video_output_path: str = ""


class EdgeVisionPipeline:
    """Main pipeline execution coordinator."""

    def __init__(self, config: EdgeVisionConfig) -> None:
        """Initialize pipeline with validated configuration."""
        self.config = config
        self.anchors = self._load_anchors(config.model.anchors_path)
        self.preprocessor = Preprocessor(input_size=config.model.input_size)
        self._detector: Optional[Any] = None

    def _load_anchors(self, anchors_path: str | Path) -> Optional[np.ndarray]:
        """Load precomputed anchor boxes if available."""
        p = Path(anchors_path)
        candidates = [
            p,
            Path("models/anchors.npy"),
            Path("edgevision/models/anchors.npy"),
            Path(__file__).resolve().parent.parent / "models" / "anchors.npy",
        ]
        for cand in candidates:
            if cand.is_file():
                return np.load(cand)
        return None

    def _resolve_model_path(self, runtime_name: str) -> Path:
        """Resolve actual model file path based on runtime mode."""
        if runtime_name in ("yolov11", "yolov11n"):
            yolo_cfg_path = getattr(self.config.model, "yolo_path", "models/yolo11n.pt")
            candidates = [
                Path(yolo_cfg_path),
                Path("models/yolo11n.pt"),
                Path("edgevision/models/yolo11n.pt"),
                Path(__file__).resolve().parent.parent / "models" / "yolo11n.pt",
                Path("yolo11n.pt"),
            ]
            for cand in candidates:
                if cand.is_file():
                    return cand
            return Path(yolo_cfg_path)

        base_path = Path(self.config.model.pte_path)
        if runtime_name == "executorch-int8":
            candidates = [
                base_path.with_name(f"{base_path.stem}_int8.pte"),
                Path("models/ssdlite320_mobilenet_v3_large_int8.pte"),
                Path("edgevision/models/ssdlite320_mobilenet_v3_large_int8.pte"),
                base_path,
            ]
            for cand in candidates:
                if cand.is_file():
                    return cand
        return base_path

    def _get_detector(self, runtime_name: str) -> Any:
        """Instantiate detector wrapper based on runtime."""
        if runtime_name in ("yolov11", "yolov11n"):
            from edgevision.detector_yolo import YOLOv11Detector
            resolved_yolo = self._resolve_model_path(runtime_name)
            return YOLOv11Detector(
                model_path=resolved_yolo,
                conf_threshold=self.config.conf_threshold,
                nms_iou=self.config.nms_iou,
            )
        elif runtime_name == "pytorch-reference":
            m_name = (
                self.config.model.name
                if self.config.model.name == "ssdlite320_mobilenet_v3_large"
                else "ssdlite320_mobilenet_v3_large"
            )
            return PyTorchDetector(model_name=m_name)
        elif runtime_name in ("executorch-fp32", "executorch-int8"):
            from edgevision.detector_et import ExecuTorchDetector
            resolved_pte = self._resolve_model_path(runtime_name)
            return ExecuTorchDetector(
                pte_path=resolved_pte,
                runtime_mode=runtime_name,
                threads=self.config.threads,
            )
        else:
            raise ValueError(f"Unknown runtime: {runtime_name}")

    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        runtime_override: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        use_threading_pipeline: bool = True,
    ) -> PipelineResult:
        """Execute complete offline processing on the target video file.

        Args:
            video_path: Path to target input video.
            output_dir: Directory where outputs and artifacts will be saved.
            runtime_override: Optional runtime selection to override config.
            progress_callback: Optional callback(processed_frames, total_frames, fps).
            use_threading_pipeline: Whether to use the two-thread decoder queue.

        Returns:
            PipelineResult container with counts, benchmarks, and artifacts.
        """
        runtime = runtime_override or self.config.model.runtime
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        meta = get_video_metadata(video_path)
        detector = self._get_detector(runtime)
        active_id_map: Dict[int, str] = getattr(
            detector, "id_to_class_map", self.config.classes.coco_id_map
        )

        # 1. Warm-up: 5 dummy inferences (excluded from stats)
        if hasattr(detector, "warmup"):
            detector.warmup(num_iterations=5, input_size=self.config.model.input_size)
        else:
            dummy_input = np.zeros(
                (1, 3, self.config.model.input_size, self.config.model.input_size),
                dtype=np.float32,
            )
            for _ in range(5):
                detector.predict_raw(dummy_input)

        # Record initial system telemetry
        temp_start = read_cpu_temperature()

        # Resolve initial stride
        is_auto_stride = (self.config.frame_stride == "auto")
        current_stride = 2 if is_auto_stride else int(self.config.frame_stride)

        # Storage for records and latencies
        per_frame_records: List[Dict[str, Any]] = []
        inference_latencies: List[float] = []
        frames_to_annotate: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]] = []

        total_frames = meta.total_frames
        fps = meta.fps
        processed_count = 0
        skipped_count = 0

        last_counts: Dict[str, int] = {name: 0 for name in self.config.classes.target_names}
        last_zone_veh: Optional[int] = 0 if self.config.zone.enabled else None
        last_zone_ped: Optional[int] = 0 if self.config.zone.enabled else None
        last_boxes = np.empty((0, 4), dtype=np.float32)
        last_scores = np.empty(0, dtype=np.float32)
        last_labels = np.empty(0, dtype=np.int64)

        # Candidate sampling interval
        sample_interval = max(1, total_frames // max(1, self.config.output.save_samples))
        sample_indices_target = {i * sample_interval for i in range(self.config.output.save_samples)}

        # Optional annotated output video writer
        save_video_enabled = bool(getattr(self.config.output, "save_video", False))
        video_writer: Optional[cv2.VideoWriter] = None
        video_out_path = out_path / "annotated_video.mp4"
        writer_dims: Optional[Tuple[int, int]] = None

        # 2. Setup video decoder: background thread with bounded queue (size 4)
        decoder = VideoDecoderThread(
            video_path=video_path,
            queue_size=4,
            frame_stride=1,
            max_width=640,
        )
        decoder.start()

        t_start = time.perf_counter()

        try:
            while True:
                item = decoder.get_frame(timeout=10.0)
                if item is None:
                    break

                frame_idx, timestamp_s, frame_bgr = item
                is_processed_frame = (frame_idx % current_stride == 0)

                # Initialize VideoWriter on first valid frame if enabled
                if save_video_enabled and video_writer is None and frame_bgr is not None:
                    h_v, w_v = frame_bgr.shape[:2]
                    w_v = w_v - (w_v % 2)
                    h_v = h_v - (h_v % 2)
                    writer_dims = (w_v, h_v)
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, float(fps), writer_dims)
                    except Exception as vw_init_err:
                        safe_print(f"[Warning] Could not initialize VideoWriter: {vw_init_err}")
                        video_writer = None

                if is_processed_frame:
                    t0 = time.perf_counter()
                    if hasattr(detector, "predict_detections"):
                        boxes, scores, labels = detector.predict_detections(
                            frame_bgr=frame_bgr,
                            conf_threshold=self.config.conf_threshold,
                            nms_iou=self.config.nms_iou,
                            max_detections=self.config.max_detections,
                            target_classes=self.config.classes.target_names,
                        )
                    else:
                        preprocessed = self.preprocessor.preprocess_frame(frame_bgr)
                        logits, regs = detector.predict_raw(preprocessed)

                        boxes, scores, labels = postprocess_detections(
                            logits=logits,
                            regressions=regs,
                            anchors=self.anchors,
                            conf_threshold=self.config.conf_threshold,
                            nms_iou=self.config.nms_iou,
                            max_detections=self.config.max_detections,
                            target_class_ids=self.config.classes.coco_id_map,
                            input_size=self.config.model.input_size,
                        )

                    t_inf = time.perf_counter() - t0
                    inference_latencies.append(t_inf)

                    current_counts = count_objects_in_frame(
                        class_ids=labels,
                        id_to_class_map=active_id_map,
                        target_classes=self.config.classes.target_names,
                    )

                    # Zone counting if enabled
                    cur_zone_veh: Optional[int] = None
                    cur_zone_ped: Optional[int] = None
                    if self.config.zone.enabled:
                        z_res = count_in_zone(
                            boxes=boxes,
                            labels=labels,
                            polygon=self.config.zone.polygon,
                            id_to_class_map=active_id_map,
                        )
                        cur_zone_veh = z_res["in_zone_vehicles"]
                        cur_zone_ped = z_res["in_zone_pedestrians"]
                        last_zone_veh = cur_zone_veh
                        last_zone_ped = cur_zone_ped

                    last_counts = current_counts
                    last_boxes = boxes
                    last_scores = scores
                    last_labels = labels
                    processed_count += 1

                    # Adaptive stride update in 'auto' mode after initial sample frames
                    if is_auto_stride and processed_count >= 5 and processed_count % 5 == 0:
                        recent_avg_latency = float(np.mean(inference_latencies[-5:]))
                        target_frame_time = 1.0 / max(fps, 1.0)
                        if recent_avg_latency > target_frame_time:
                            needed_stride = max(1, math.ceil(recent_avg_latency / target_frame_time))
                            current_stride = min(6, needed_stride)

                    vehicle_total = compute_vehicle_total(current_counts)
                    counts_repr = " ".join(f"{k}:{v}" for k, v in current_counts.items() if v > 0) or "none"
                    safe_print(
                        f"Frame {frame_idx:04d} ({timestamp_s:.2f}s) [{runtime}]: "
                        f"Vehicles={vehicle_total} Pedestrians={current_counts.get('person', 0)} | {counts_repr} "
                        f"({t_inf*1000:.1f}ms, stride={current_stride})"
                    )
                else:
                    skipped_count += 1
                    current_counts = dict(last_counts)
                    cur_zone_veh = last_zone_veh
                    cur_zone_ped = last_zone_ped
                    boxes = last_boxes
                    scores = last_scores
                    labels = last_labels

                # Record frame data
                frame_record = FrameCounts(
                    frame_index=frame_idx,
                    timestamp_seconds=timestamp_s,
                    counts_by_class=dict(current_counts),
                    total_vehicles=compute_vehicle_total(current_counts),
                    total_pedestrians=current_counts.get("person", 0),
                    in_zone_vehicles=cur_zone_veh,
                    in_zone_pedestrians=cur_zone_ped,
                )
                per_frame_records.append(frame_record.to_csv_dict())

                # Write frame to video if output video recording is active
                if save_video_enabled and video_writer is not None and writer_dims is not None:
                    try:
                        annotated_full = annotate_frame(
                            frame=frame_bgr,
                            boxes=boxes,
                            scores=scores,
                            class_ids=labels,
                            class_name_map=active_id_map,
                            counts=dict(current_counts),
                            zone_polygon=self.config.zone.polygon if self.config.zone.enabled else None,
                            blur_persons=self.config.privacy.blur_persons_in_samples,
                        )
                        if (annotated_full.shape[1], annotated_full.shape[0]) != writer_dims:
                            annotated_full = cv2.resize(annotated_full, writer_dims, interpolation=cv2.INTER_LINEAR)
                        video_writer.write(annotated_full)
                    except Exception as vw_err:
                        safe_print(f"[Warning] Failed to write video frame {frame_idx}: {vw_err}")

                # Capture sample candidates (capped to prevent memory growth)
                if (frame_idx in sample_indices_target or (is_processed_frame and len(frames_to_annotate) < 10)) and len(frames_to_annotate) < 30:
                    frames_to_annotate.append((
                        frame_idx,
                        frame_bgr.copy(),
                        boxes.copy(),
                        scores.copy(),
                        labels.copy(),
                        dict(current_counts),
                    ))

                if progress_callback and frame_idx % 10 == 0:
                    elapsed = time.perf_counter() - t_start
                    cur_fps = (frame_idx + 1) / max(elapsed, 0.001)
                    progress_callback(frame_idx + 1, total_frames, cur_fps)
        finally:
            decoder.stop()
            if video_writer is not None:
                video_writer.release()

        t_total = time.perf_counter() - t_start
        temp_end = read_cpu_temperature()
        peak_rss = get_peak_rss_mb()

        # 3. Apply temporal count smoothing if enabled
        if self.config.smoothing.enabled and len(per_frame_records) > 1:
            per_frame_records = apply_rolling_median_counts(
                per_frame_records,
                window_size=self.config.smoothing.window,
            )

        # 4. Compute statistics and metrics
        total_seen = processed_count + skipped_count
        e2e_fps = total_seen / max(t_total, 0.001)
        effective_fps = processed_count / max(sum(inference_latencies), 0.001) if inference_latencies else 0.0
        percentiles = calculate_latency_percentiles(inference_latencies)

        model_file = self._resolve_model_path(runtime)
        model_size_mb = (model_file.stat().st_size / (1024.0 * 1024.0)) if model_file.is_file() else None

        metrics = BenchmarkMetrics(
            runtime_used=runtime,
            num_processed_frames=processed_count,
            num_skipped_frames=skipped_count,
            total_time_seconds=t_total,
            end_to_end_fps=e2e_fps,
            effective_fps=effective_fps,
            latency_mean_ms=percentiles["mean_ms"],
            latency_p50_ms=percentiles["p50_ms"],
            latency_p95_ms=percentiles["p95_ms"],
            latency_p99_ms=percentiles["p99_ms"],
            peak_rss_mb=peak_rss,
            cpu_temp_start_c=temp_start,
            cpu_temp_end_c=temp_end,
            model_size_mb=model_size_mb,
        )

        # Save benchmark.json
        bench_json_path = out_path / "benchmark.json"
        metrics.save_json(bench_json_path)

        # Print results table
        print_benchmark_table(metrics)

        # 5. Save annotated sample frames
        saved_samples: List[str] = []
        if frames_to_annotate:
            num_samples = min(len(frames_to_annotate), max(2, self.config.output.save_samples))
            sample_indices = np.linspace(0, len(frames_to_annotate) - 1, num_samples, dtype=int)
            samples_dir = out_path / "samples"
            samples_dir.mkdir(exist_ok=True)

            for s_idx in sample_indices:
                f_idx, img, b, s, l, c = frames_to_annotate[s_idx]
                annotated = annotate_frame(
                    frame=img,
                    boxes=b,
                    scores=s,
                    class_ids=l,
                    class_name_map=active_id_map,
                    counts=c,
                    zone_polygon=self.config.zone.polygon if self.config.zone.enabled else None,
                    blur_persons=self.config.privacy.blur_persons_in_samples,
                )
                sample_file = samples_dir / f"sample_frame_{f_idx:04d}.jpg"
                cv2.imwrite(str(sample_file), annotated)
                saved_samples.append(str(sample_file))

        # 6. Write buffered CSV at end of run
        csv_file = out_path / "counts.csv"
        if per_frame_records:
            fieldnames = list(per_frame_records[0].keys())
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_frame_records)

        # 7. Peak and average counts
        peak_counts: Dict[str, int] = {}
        avg_counts: Dict[str, float] = {}
        for cls_name in self.config.classes.target_names:
            vals = [int(r.get(cls_name, 0)) for r in per_frame_records]
            peak_counts[cls_name] = max(vals) if vals else 0
            avg_counts[cls_name] = float(np.mean(vals)) if vals else 0.0

        vehicle_vals = [int(r.get("total_vehicles", 0)) for r in per_frame_records]
        peak_counts["vehicles"] = max(vehicle_vals) if vehicle_vals else 0
        avg_counts["vehicles"] = float(np.mean(vehicle_vals)) if vehicle_vals else 0.0

        # Congestion classification
        congestion_level = determine_congestion_level(
            average_vehicles=avg_counts["vehicles"],
            low_max=float(self.config.congestion.low_max),
            medium_max=float(self.config.congestion.medium_max),
        )

        # 8. Render trend chart PNG and summary markdown
        chart_file = out_path / "counts_chart.png"
        plot_counts_over_time(per_frame_records, chart_file)

        summary_file = out_path / "summary.md"
        summary_payload = {
            "peak_counts": peak_counts,
            "average_counts": avg_counts,
            "congestion_level": congestion_level,
            "runtime_used": runtime,
            "total_frames": total_frames,
            "processed_frames": processed_count,
            "end_to_end_fps": round(e2e_fps, 2),
            "processing_time_seconds": round(t_total, 3),
            "benchmark_metrics": metrics,
        }
        generate_summary_markdown(summary_payload, summary_file)

        # 9. Failure-case export if ground truth exists
        gt_candidates = [
            Path("ground_truth/ground_truth.csv"),
            out_path / "ground_truth.csv",
        ]
        for gt_c in gt_candidates:
            if gt_c.is_file():
                export_failure_cases(per_frame_records, gt_c, video_path, out_path, top_k=3)
                break

        return PipelineResult(
            video_path=str(video_path),
            runtime_used=runtime,
            total_frames=total_frames,
            processed_frames=processed_count,
            skipped_frames=skipped_count,
            fps_video=fps,
            effective_fps=round(effective_fps, 2),
            end_to_end_fps=round(e2e_fps, 2),
            processing_time_seconds=round(t_total, 3),
            per_frame_counts=per_frame_records,
            peak_counts=peak_counts,
            average_counts=avg_counts,
            congestion_level=congestion_level,
            benchmark_metrics=metrics,
            saved_sample_paths=saved_samples,
            output_dir=str(out_path),
            csv_path=str(csv_file),
            benchmark_json_path=str(bench_json_path),
            summary_md_path=str(summary_file),
            chart_path=str(chart_file),
            video_output_path=str(video_out_path) if save_video_enabled and video_out_path.is_file() else "",
        )
