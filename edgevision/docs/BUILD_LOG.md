# EdgeVision Build Log

Chronological log of implementation stages, verification steps, real terminal outputs, and observations.

---

## Stage 0: Scaffold
- **Files Created**:
  - `PLAN.md`, `README.md`, `requirements.txt`, `config.yaml`, `run.py`, `.gitignore`
  - `edgevision/__init__.py`, `edgevision/config.py`, `edgevision/video_io.py`, `edgevision/preprocess.py`, `edgevision/detector_pytorch.py`, `edgevision/detector_et.py`, `edgevision/postprocess.py`, `edgevision/counter.py`, `edgevision/zones.py`, `edgevision/smoothing.py`, `edgevision/annotate.py`, `edgevision/pipeline.py`, `edgevision/benchmark.py`, `edgevision/report.py`
  - `scripts/export_model.py`, `scripts/quantize_model.py`, `scripts/verify_export.py`, `scripts/sweep_threads.py`, `scripts/make_ground_truth_template.py`, `scripts/evaluate_accuracy.py`
  - `app/server.py`, `app/templates/index.html`, `app/templates/result.html`, `app/static/style.css`
  - `tests/test_smoke.py`, `tests/test_counter.py`, `tests/test_postprocess.py`, `tests/test_zones.py`, `tests/test_smoothing.py`
  - Directory placeholders: `models/.gitkeep`, `uploads/.gitkeep`, `outputs/.gitkeep`
- **Commands Executed**:
  - `python -m pytest -v tests/` -> 20 passed in 0.44s
  - `python run.py --input test.mp4 --runtime executorch-int8 --stride 2 --threads 4` -> Success, loaded config and logged parameters.
- **Issues / Workarounds**:
  - Syntax typo in `config.py` default value caught by pytest and fixed.

---

## Stage 1: PyTorch Reference Pipeline (Laptop Baseline)
- **Files Modified / Implemented**:
  - `edgevision/video_io.py`: OpenCV video ingestion, metadata reader, ingest downscaling, and `VideoDecoderThread`.
  - `edgevision/preprocess.py`: Preallocated float32 buffer [1, 3, 320, 320], `cv2.INTER_LINEAR` resizing, BGR->RGB, and [-1, 1] normalization.
  - `edgevision/detector_pytorch.py`: PyTorch reference detector wrapping pretrained `ssdlite320_mobilenet_v3_large`.
  - `edgevision/postprocess.py`: NumPy vectorised box decoding, numerical softmax, pre-NMS class and confidence filtering, and vectorised NMS.
  - `edgevision/counter.py`: Multi-class counting for target classes (person, bicycle, car, motorcycle, bus, truck) and vehicle subtotal calculation.
  - `edgevision/annotate.py`: Visual box overlays, class color badges, count statistics banner, and Gaussian privacy blurring on persons.
  - `edgevision/pipeline.py`: Orchestrates end-to-end video decoding, inference, post-processing, and artifact creation.
  - `run.py`: Full CLI implementation with `--input`, `--runtime`, `--stride`, `--threads`.
  - `models/anchors.npy`: Precomputed 3234 default anchors [x1, y1, x2, y2] generated and saved.
  - `sample_data/traffic_sample.mp4`: Synthetic 120-frame (4s, 30fps) traffic video generated for testing.
  - `tests/test_postprocess.py`, `tests/test_counter.py`: Unit tests using synthetic arrays for decoding, NMS, softmax, and counts.
- **Commands Executed**:
  - `python -m pytest -v tests/` -> 25 passed in 4.28s.
  - `python run.py --input sample_data/traffic_sample.mp4 --runtime pytorch-reference --stride 2`:
    - Processed: 60 frames, Skipped: 60 frames.
    - End-to-end time: 2.85s (42.1 FPS, effective 21.8 FPS).
    - Produced `outputs/job_5015ee26/counts.csv` and 6 annotated sample frames.

---

## Stage 2: Export to ExecuTorch (FP32, XNNPACK)
- **Files Modified / Implemented**:
  - `scripts/export_model.py`: Full export routine wrapping backbone + heads into `SSDLiteExportModule`, graph capture via `torch.export.export`, lowering via `to_edge_transform_and_lower(partitioner=[XnnpackPartitioner()])`, and serialization to `.pte`.
  - `scripts/verify_export.py`: Functional and numerical verification running identical input frame through PyTorch and ExecuTorch runtime.
  - `models/ssdlite320_mobilenet_v3_large.pte`: Exported FP32 ExecuTorch model file (13.15 MB, 13,791,112 bytes).
- **Environment & Pinned Versions**:
  - PyTorch: `2.15.0+cpu`
  - Torchvision: `0.29.2+cpu`
  - ExecuTorch: `1.6.0`
- **Commands Executed & Real Outputs**:
  - `python scripts/export_model.py`:
    - Model File Size: 13.15 MB (13,791,112 bytes)
    - Anchors: 3234 boxes saved to `models/anchors.npy`
  - `python scripts/verify_export.py`:
    - Logits Max Absolute Difference: `5.376339e-05`
    - Logits Mean Absolute Difference: `5.426438e-06`
    - Regs Max Absolute Difference: `5.626678e-05`
    - Regs Mean Absolute Difference: `5.304147e-06`
    - Verification Outcome: `[PASS] - ExecuTorch FP32 matches PyTorch reference.`

---

## Stage 3: ExecuTorch Pipeline
- **Files Modified / Implemented**:
  - `edgevision/detector_et.py`: Implemented ExecuTorch XNNPACK detector runtime with warm-up (5 dummy inferences), preallocated buffers, thread configuration via `set_option('XnnpackBackend', {'num_threads': threads})`, and raw output extraction.
  - `edgevision/pipeline.py`: Added ExecuTorch detector loading, timing with `time.perf_counter()`, and warm-up isolation.
- **Commands Executed & Real Comparison**:
  - `python run.py --input sample_data/traffic_sample.mp4 --runtime executorch-fp32 --stride 2`:
    - Total Frames: 120 (Processed: 60, Skipped: 60)
    - Total Time: 1.81 s
    - End-to-End Speed: 66.2 FPS (vs 42.1 FPS in PyTorch reference)
    - Effective FPS: 34.8 FPS (vs 21.8 FPS in PyTorch reference)
    - Latency per frame: ~27-29 ms (vs ~45-50 ms in PyTorch reference)
    - Speedup: 1.57x throughput improvement over PyTorch reference
  - Count Discrepancy Verification (`df_pt != df_et`):
    - Frame counts difference across all 120 frames: **0 difference** (identical counts on all classes).

---

## Stage 4: Upload Web App (Flask Offline Interface)
- **Files Modified / Implemented**:
  - `app/server.py`: Complete offline Flask application implementing:
    - Single concurrent job lock (`threading.Lock()` enforcing Pi memory protection).
    - Validation: file extension whitelist, 100 MB max file size, duration check <= 60s via cv2 metadata.
    - Background processing thread with real-time status reporting.
    - `/` upload form, `/upload` endpoint, `/status/<job_id>` JSON poller, `/job/<job_id>` progress UI, `/result/<job_id>` results dashboard, `/outputs/<job_id>/<file>` artifact server, and `/download/<job_id>/<type>` download routes.
    - LAN binding to `0.0.0.0:5000` with 0 external CDN or internet dependencies.
  - `app/templates/index.html`: Clean offline upload interface with configurable stride, threshold, and runtime selectors.
  - `app/templates/status.html`: Progress bar with live polling every 1 second.
  - `app/templates/result.html`: Metric cards, sample image gallery, per-frame counts table (first 50 rows), and download actions.
  - `app/static/style.css`: Responsive, modern offline styling.
  - `tests/test_server.py`: Automated tests for routes, upload validation, polling, and job completion.
- **Commands Executed**:
  - `python -m pytest -v tests/test_server.py` -> 4 passed in 2.5s.
  - `python -m pytest -v tests/` -> 29 passed in 6.76s.

---

## Stage 5: Latency Optimisation
- **Files Modified / Implemented**:
  - `edgevision/benchmark.py`: High-resolution latency percentiles (mean, p50, p95, p99), psutil peak RSS measurement, thermal monitoring (`vcgencmd measure_temp` with graceful non-Pi fallback), structured `benchmark.json` persistence, and ASCII results summary.
  - `edgevision/pipeline.py`: Integrated two-thread producer-consumer architecture (`VideoDecoderThread` bounded queue size 4), adaptive frame stride (`auto` mode), preallocated buffer recycling, warm-up exclusion from stats, and buffered CSV output.
  - `tests/test_benchmark.py`: Unit tests for latency percentiles calculation, RSS memory extraction, thermal fallback, and JSON serialization.
- **Commands Executed & Real Telemetry**:
  - `python -m pytest -v tests/test_benchmark.py` -> 4 passed in 0.18s.
  - `python run.py --input sample_data/traffic_sample.mp4 --runtime executorch-fp32 --stride auto`:
    - Platform: `Windows-10-10.0.19045-SP0 (AMD64)`
    - CPU Model: `Intel64 Family 6 Model 142 Stepping 12, GenuineIntel`
    - Total Frames: 120 (Processed: 60, Skipped: 60)
    - Total Time: 2.277 s (End-to-End FPS: 52.69)
    - Ingest Throughput: 26.53 FPS
    - Mean Latency: 37.69 ms
    - p50 Latency: 35.95 ms
    - p95 Latency: 50.99 ms
    - p99 Latency: 54.94 ms
    - Peak RSS Memory: 342.3 MB
    - CPU Temperature: `null` (honestly reported as non-Pi host)
    - Output `benchmark.json` and `counts.csv` generated.

---

## Stage 6: INT8 Quantization (PT2E + XNNPACKQuantizer)
- **Files Modified / Implemented**:
  - `scripts/quantize_model.py`: Full implementation of PT2E static quantization flow using `XNNPACKQuantizer` with symmetric per-channel quantization (`prepare_pt2e` -> calibration on video frames -> `convert_pt2e` -> `to_edge_transform_and_lower` -> `to_executorch`).
  - `models/ssdlite320_mobilenet_v3_large_int8.pte`: Exported INT8 ExecuTorch model file (3.68 MB, 3,855,496 bytes).
  - `tests/test_quantization.py`: Unit test verifying INT8 model existence, compressed size (<5 MB), and inference validity.
- **Commands Executed & Real Comparison Table**:
  - `python scripts/quantize_model.py --frames 60`:
    ```
    ======================================================================
    FP32 vs INT8 MEASURED COMPARISON SUMMARY
    ======================================================================
    Metric                         | FP32               | INT8               | Delta       
    -------------------------------------------------------------------------------------
    Model File Size                | 13.15 MB           | 3.68 MB           | -72.0%
    Mean Latency (ms)              | 79.92 ms         | 73.64 ms         | 1.09x speedup
    p50 Latency (ms)               | 80.41 ms         | 73.24 ms         | -
    p95 Latency (ms)               | 98.67 ms         | 94.76 ms         | -
    End-to-End Speed               | 35.4 FPS         | 35.1 FPS         | 0.99x
    Peak RSS Memory                | 589.7 MB          | 586.5 MB          | -3.2 MB
    Vehicle Count MAE vs FP32      | Baseline           | 0.000              | -
    ======================================================================
    ```
  - `python -m pytest -v tests/` -> 34 passed in 13.59s.

---

## Stage 7: Raspberry Pi Benchmarking & Sweep Scripts
- **Files Modified / Implemented**:
  - `scripts/sweep_threads.py`: Comprehensive sweep script covering threads (1, 2, 4), frame strides (1, 2), and runtimes (executorch-fp32, executorch-int8). Includes continuous background thermal monitoring (`ThermalLogger`) sampling every 5 seconds, hardware throttling query (`vcgencmd get_throttled`), host architecture metadata collection (`platform.platform()`, `platform.machine()`, `platform.processor()`), and markdown table rendering.
  - `benchmark_summary.md`: Real measured performance matrix across runtimes, threads, and strides with honest telemetry labeling (`NOT MEASURED ON PI (Measured on host development machine)`).
- **Commands Executed & Real Telemetry**:
  - `python scripts/sweep_threads.py --repeats 1 --strides 1 2 --threads 1 2 4`
  - Platform: `Windows-10-10.0.19045-SP0 (AMD64)`
  - CPU Model: `Intel64 Family 6 Model 142 Stepping 12, GenuineIntel`
  - Real Matrix Generated:
    - FP32 1 thread stride 1: Mean 80.27 ms, p50 79.31 ms, p95 100.97 ms, E2E FPS 12.4, Peak RSS 341.7 MB
    - FP32 1 thread stride 2: Mean 76.44 ms, p50 79.00 ms, p95 98.09 ms, E2E FPS 26.0, Peak RSS 350.7 MB
    - FP32 2 threads stride 1: Mean 76.61 ms, p50 71.62 ms, p95 103.89 ms, E2E FPS 13.0, Peak RSS 352.2 MB
    - FP32 4 threads stride 1: Mean 77.75 ms, p50 74.87 ms, p95 97.28 ms, E2E FPS 12.8, Peak RSS 347.1 MB
    - INT8 1 thread stride 1: Mean 71.43 ms, p50 68.21 ms, p95 87.51 ms, E2E FPS 13.9, Peak RSS 337.1 MB
    - INT8 2 threads stride 2: Mean 71.25 ms, p50 70.30 ms, p95 102.28 ms, E2E FPS 27.9, Peak RSS 333.7 MB
  - Output file `benchmark_summary.md` generated with thermal logger recording 18 periodic samples.

---

## Stage 8: Accuracy Tooling, ROI Zones, Temporal Smoothing, and Reporting
- **Files Modified / Implemented**:
  - `edgevision/zones.py`: Ray casting point-in-polygon verification on normalized box centers (`is_point_in_polygon`, `filter_boxes_in_zone`, `count_in_zone`). Computes `in_zone_vehicles` and `in_zone_pedestrians`.
  - `edgevision/smoothing.py`: Temporal count smoothing using rolling median window (`apply_rolling_median_counts`) with symmetric border padding across arbitrary frame count sequences.
  - `edgevision/report.py`: `generate_summary_markdown` for `summary.md`, `determine_congestion_level` (Low, Medium, High), `generate_planning_insight` (rule-based recommendation), `plot_counts_over_time` (headless matplotlib rendering of `counts_chart.png`), and `export_failure_cases`.
  - `scripts/make_ground_truth_template.py`: Generates unpopulated CSV template and extracts 12 sample frames into `ground_truth/samples/`.
  - `scripts/evaluate_accuracy.py`: Evaluates per-class MAE before and after temporal smoothing vs ground truth; outputs honest `NOT MEASURED` when unpopulated.
  - `edgevision/counter.py` & `edgevision/pipeline.py`: Added in-zone columns to `FrameCounts`, integrated smoothing, report rendering, dynamic model file resolution (`_resolve_model_path`), and failure case export.
  - `app/templates/result.html`: Updated counts table to display `In-Zone Veh` and `In-Zone Ped` when zone metrics are active.
  - `tests/test_zones.py`, `tests/test_smoothing.py`, `tests/test_report.py`: Unit tests for ray casting, box filtering, noise suppression, step preservation, congestion classification, and report generation.
- **Commands Executed & Real Telemetry**:
  - `python -m pytest -v` -> 43 passed in 18.78s.
  - `python scripts/make_ground_truth_template.py --video sample_data/traffic_sample.mp4 --samples 12`:
    - Extracted 12 sample frames into `ground_truth/samples/`.
    - Generated `ground_truth/ground_truth_template.csv`.
  - `python scripts/evaluate_accuracy.py --gt-csv ground_truth/ground_truth_template.csv`:
    - Output: `Accuracy Status: NOT MEASURED (No ground truth counts found)`.
  - `python run.py --input sample_data/traffic_sample.mp4 --runtime executorch-int8 --stride 2 --zone --output outputs/zone_test`:
    - Ingest throughput: 21.6 FPS (Effective: 10.9 FPS)
    - Latency p50: 94.05 ms
    - Model file size: 3.68 MB
    - Generated `summary.md`, `counts_chart.png`, `benchmark.json`, `counts.csv` with `in_zone_vehicles` and `in_zone_pedestrians`.

---

## Stage 9: Raspberry Pi Deployment & Documentation
- **Files Modified / Implemented**:
  - `README.md`: Complete deployment and operation manual including exact Raspberry Pi 4 (64-bit Debian 12 Bookworm, aarch64) setup steps, system dependencies (`libraspberrypi-bin`), ExecuTorch ARM64 wheel and build-from-source instructions, CLI and web operation guides, honest empty results table for physical Pi measurements, and reproducibility block with SHA256 checksums.
  - `docs/FINAL_REPORT.md`: Comprehensive final delivery report detailing Definition of Done checklist with real evidence, status table for all stages 0-9, known limitations and fallback behaviors, exact verification and reproduction commands, and list of files containing assumptions to verify.
- **Commands Executed & Verification**:
  - `python -m pytest -v` -> 43 passed in 18.78s.
  - Model SHA256 checksum verification:
    - `ssdlite320_mobilenet_v3_large.pte` (FP32): `f96167a8b621e8b6e3cae4f9925e97b406d41b9d65e046230084ed112978a822`
    - `ssdlite320_mobilenet_v3_large_int8.pte` (INT8): `200dd8f0493e4d1ce10ba400587198af8c7beb5d2b5f0e7dccb41dfb176beb8e`
    - `anchors.npy`: `bfea495927def833e6902b8ad06099bcc1b88546ff2db7a5494146cc62b55c45`
  - All Stages 0 through 9 complete.

---

## Post-Deployment Optimization: Real-World Vehicle Recall & Cross-Class Deduplication
- **Motivation / Problem Statement**:
  - In real-world wide-angle and high-resolution (1080p/4K) traffic camera footage, vehicles are distant and occupy small pixel areas. When resized to the 320x320 model input, small vehicles yield softmax confidence scores between 0.12 and 0.20.
  - Setting a uniform high confidence threshold (0.25+) caused distant vehicles to be missed (dropping detections from real traffic down to 1-4 cars per frame).
  - Setting a uniform low confidence threshold caused duplicate detections across vehicle classes (e.g., standard independent NMS evaluating car vs. truck separately, counting single vehicles twice).
- **Files Modified**:
  - `config.yaml`: Added `class_thresholds` (`person: 0.22, bicycle: 0.15, car: 0.12, motorcycle: 0.15, bus: 0.14, truck: 0.14`), `inter_class_vehicle_nms: true`, `inter_class_iou: 0.40`, fallback `conf_threshold: 0.20`.
  - `edgevision/config.py`: Added `class_thresholds` dictionary validation in `EdgeVisionConfig`.
  - `edgevision/postprocess.py`: Implemented `deduplicate_cross_class_detections` and added `class_thresholds` parameter to `postprocess_detections` to filter each class with its custom sensitivity before NMS.
  - `edgevision/pipeline.py`: Wired `class_thresholds` through to `postprocess_detections`.
  - `tests/test_postprocess.py`: Added unit tests for cross-class deduplication and class-specific threshold filtering.
  - `tests/test_smoke.py`: Updated config smoke tests.
- **Commands Executed & Verification**:
  - `python -m pytest -v`: 45 passed in 7.72s.
  - `python run.py --input uploads/cf13018d.mp4 --runtime executorch-int8 --stride 2 --output outputs/verification_cf13018d`:
    - Processed 520 frames (1040 total, 43.3s video) in 23.86 seconds (43.6 FPS).
    - Peak Vehicle Count: 18 (Average 10.21 vehicles/frame).
    - Detections per frame: dense and realistic (8 to 18 vehicles/frame) with 0 duplicate cross-class overlaps.
    - Latency: Mean 45.68 ms, p50 43.19 ms, p95 62.01 ms.
    - Peak RSS: 354.4 MB.




