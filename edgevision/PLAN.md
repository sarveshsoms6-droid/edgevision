# ROLE
You are a senior embedded-ML engineer. Build "EdgeVision", an offline vehicle and pedestrian detection and counting system that runs on a Raspberry Pi 4 (4 GB, 64-bit Raspberry Pi OS) using ExecuTorch (XNNPACK backend), with a minimal web page to upload a traffic video.

# HARD RULES (read before every stage)
1. Work in STAGES. Do ONLY the stage I name. At the end of each stage: list files created, show the exact commands to verify, run them if you can, and STOP and wait.
2. NEVER invent benchmark numbers, accuracy figures or sample outputs. Every number must come from code that actually ran. If something was not measured, print "NOT MEASURED".
3. ExecuTorch APIs change between releases. Before writing export or runtime code, consult the OFFICIAL ExecuTorch documentation for the installed version, print `executorch.__version__` and `torch.__version__`, and pin both in requirements.txt. Do not guess function names. Likely entry points to verify (do not trust blindly): `torch.export.export`, `executorch.exir.to_edge_transform_and_lower`, `executorch.backends.xnnpack.partition.xnnpack_partitioner.XnnpackPartitioner`, `executorch.runtime.Runtime`.
4. Do NOT use cloud APIs, internet calls at runtime, object tracking IDs, live camera streaming, or databases. Everything runs offline and locally.
5. Do NOT silently fall back to PyTorch inference. Every result file must record which runtime produced it ("executorch-fp32", "executorch-int8", or "pytorch-reference"). Pi benchmarks must use ExecuTorch only.
6. No training. Use a pretrained COCO detector. The model must be swappable via config.yaml (default: torchvision ssdlite320_mobilenet_v3_large, COCO weights). Note in README that the exact model must be confirmed against the organisers' approved list.
7. Keep code simple, typed, commented, and testable. Python 3.10+. No heavy dependencies (no TensorFlow, no ultralytics unless I ask).

# PROJECT GOAL
Input: an uploaded traffic video, up to 60 seconds, low resolution (e.g. 360p to 720p).
Output per video:
- Per-frame counts (terminal log and CSV): person, bicycle, car, motorcycle, bus, truck
- Annotated sample frames (boxes plus per-class count overlay)
- benchmark.json (latency p50, p95, mean, FPS, model size, peak RSS memory, CPU temperature, runtime used)
- summary.md (peak counts, averages, vehicle-to-pedestrian ratio, congestion label, count-over-time chart PNG, one plain-language planning insight)
- Optional: zone (ROI) counts, and an annotated output video only if enabled in config.

# PERFORMANCE TARGET ("near-real-time", not "zero latency")
- Primary: total processing time for a 60 s clip <= 60 s on Pi 4 (throughput >= clip FPS after frame skipping).
- Report honestly: mean, p50, p95, p99 inference latency per processed frame, end-to-end FPS, effective FPS including skipped frames, peak RAM, CPU temperature at start and end.
- Never claim zero latency. If the target is missed, report the real numbers and list what you tried.

# LATENCY STRATEGY (implement all, each toggleable in config.yaml)
1. Model input 320x320 by default (configurable). Resize with cv2.INTER_LINEAR. Preallocate buffers, avoid per-frame allocations.
2. INT8 post-training quantization (PT2E, XNNPACKQuantizer) with calibration on 100 to 200 frames from the uploaded or sample video.
3. XNNPACK thread count configurable (default 4). Provide a sweep script for 1 to 4 threads.
4. Two-thread pipeline: a decoder thread reads and preprocesses frames into a bounded queue (size 4); the inference thread consumes. Never let the queue grow unbounded.
5. Adaptive frame skipping: process every Nth frame (config `frame_stride`, default 2). Skipped frames reuse the last counts (NO tracking). Add "auto" mode that picks stride so processing keeps up with the video FPS.
6. Vectorised NumPy post-processing: box decoding and NMS written without Python per-box loops where possible. Class filter and confidence threshold applied BEFORE NMS.
7. Do not write annotated images for every frame. Save only N evenly spaced samples (config `save_samples`, default 6) plus the frames with the highest counts.
8. Warm-up: run 5 dummy inferences before timing. Exclude warm-up from stats.
9. Write CSV once at the end (or buffered), not per frame.
10. Ingest downscale: if the uploaded video is wider than 640 px, downscale it once with OpenCV while reading (do not re-encode a copy unless needed).

# MODEL EXPORT DESIGN (important)
Detection post-processing (anchor decoding, NMS) often fails to export. Therefore:
- Export ONLY the neural network (backbone plus heads) as an nn.Module that takes a float tensor [1,3,H,W] (already normalised) and returns raw tensors (class logits and box regressions, concatenated).
- Precompute the anchors once and save them as anchors.npy next to the .pte.
- Implement box decoding, softmax or sigmoid, thresholding, class filtering and NMS in NumPy in `postprocess.py`.
- Keep normalisation and resize in NumPy/OpenCV, not inside the exported graph.
- scripts/verify_export.py must run the same frame through PyTorch and through the .pte and report max absolute difference of raw outputs and whether top detections match.
- If export fails, print the real error, document it in docs/EXPORT_ISSUES.md, and propose alternatives (a different approved model). Do not hide the failure.

# FOLDER STRUCTURE
```
edgevision/
  PLAN.md                 # copy of this prompt
  README.md
  requirements.txt        # pinned versions
  config.yaml
  run.py                  # CLI: python run.py --input clip.mp4 --runtime executorch-int8
  edgevision/
    __init__.py
    config.py             # load/validate YAML
    video_io.py           # open video, metadata, duration check, decoder thread
    preprocess.py
    detector_pytorch.py   # reference only (laptop validation)
    detector_et.py        # ExecuTorch runtime wrapper (load .pte, run, warm-up)
    postprocess.py        # decode, filter, NMS (NumPy)
    counter.py            # per-class counts, vehicle total
    zones.py              # polygon ROI counting (point-in-polygon on box centre)
    smoothing.py          # rolling median over counts
    annotate.py           # boxes, labels, per-class overlay, optional person blur
    pipeline.py           # orchestrates everything, returns results object
    benchmark.py          # latency stats, psutil RAM, vcgencmd temp (graceful on non-Pi)
    report.py             # summary.md, chart PNG, congestion label
  scripts/
    export_model.py       # fp32 .pte
    quantize_model.py     # int8 .pte (PT2E)
    verify_export.py
    sweep_threads.py
    make_ground_truth_template.py  # CSV template for 10-15 sampled frames
    evaluate_accuracy.py  # MAE per class vs manual counts, before/after smoothing
  app/
    server.py             # Flask upload app
    templates/index.html
    templates/result.html
    static/style.css
  models/                 # .pte, anchors.npy (gitignored if large)
  uploads/                # gitignored
  outputs/                # one subfolder per job id
  tests/
    test_postprocess.py
    test_zones.py
    test_smoothing.py
    test_counter.py
```

# CONFIG (config.yaml, all keys documented with comments)
model: {name, pte_path, anchors_path, input_size: 320, runtime: executorch-int8}
classes: person, bicycle, car, motorcycle, bus, truck (use COCO ids; map correctly)
conf_threshold: 0.40, nms_iou: 0.45, max_detections: 100
frame_stride: auto (or int), threads: 4
smoothing: {enabled: true, window: 5}
zone: {enabled: false, polygon: [[x,y],...] normalised 0-1 coordinates}
congestion: {low_max: 5, medium_max: 12}  # vehicles per frame thresholds, editable
privacy: {blur_persons_in_samples: true}
output: {save_samples: 6, save_video: false, csv: true}
upload: {max_seconds: 60, max_mb: 100, allowed: [mp4, avi, mov, mkv]}

# STAGES

## STAGE 0: Scaffold
Create the folder structure, requirements.txt (do not pin ExecuTorch yet, mark TODO), config.yaml with comments, config loader with validation, empty module files with docstrings, and a README skeleton. Add .gitignore. Run a smoke test that imports every module. STOP.

## STAGE 1: PyTorch reference pipeline (laptop)
Implement video_io, preprocess, detector_pytorch, postprocess, counter, annotate and run.py using the PyTorch model. Process a video, print per-frame counts in the terminal, write counts CSV, save 2+ annotated frames. Add unit tests for postprocess and counter using synthetic arrays. This is only the correctness baseline. STOP.

## STAGE 2: Export to ExecuTorch
Implement export_model.py (fp32 .pte, XNNPACK) and verify_export.py. Follow the MODEL EXPORT DESIGN. Print versions. Report model file size and the max output difference vs PyTorch. If it fails, follow the failure rule. STOP.

## STAGE 3: ExecuTorch pipeline
Implement detector_et.py and make run.py support --runtime executorch-fp32. Re-run the clip and compare counts with the PyTorch reference (report the difference). Add warm-up and timing with time.perf_counter. STOP.

## STAGE 4: Upload web app
Implement app/server.py with Flask:
- GET / : simple page with a file input, "Upload and Analyse" button, a stride/threshold options section (defaults from config), and a note "Max 60 seconds, 100 MB".
- POST /upload : validate extension (allowlist), size (MAX_CONTENT_LENGTH), and duration using cv2 (reject > 60 s with a clear message). Use werkzeug secure_filename and a uuid job id. Save to uploads/<job_id>.<ext>.
- Run processing in a background thread. Allow ONLY ONE running job at a time (Pi memory); a second upload gets "busy" and is told to wait.
- GET /status/<job_id> : JSON {state, processed_frames, total_frames, current_fps}. The page polls it every second and shows a progress bar.
- GET /result/<job_id> : shows counts table (first 50 rows plus totals), sample annotated images, benchmark table, summary.md rendered as text, count-over-time chart, and download links for CSV and benchmark.json.
- Bind to 0.0.0.0:5000 so I can open http://<pi-ip>:5000 from a laptop on the same Wi-Fi. LAN only; no auth needed, but note this in README.
- Keep HTML minimal and clean (no frameworks, no CDN calls, no JS libraries; it must work offline).
Also keep the CLI working. STOP.

## STAGE 5: Latency optimisation
Implement the decoder/inference two-thread pipeline, adaptive frame stride, preallocated buffers and the other items in LATENCY STRATEGY. Add benchmark.py with p50/p95/p99, psutil peak RSS, and vcgencmd measure_temp (return None gracefully if unavailable). Print a clear results table at the end of every run. STOP.

## STAGE 6: INT8 quantization
Implement quantize_model.py using PT2E post-training quantization with XNNPACKQuantizer and calibration on 100 to 200 frames from a sample video. Export int8 .pte. Compare fp32 vs int8: latency, size, RAM, and count difference on the same clip. If quantization hurts accuracy badly, report it honestly. STOP.

## STAGE 7: Pi benchmarking
Write sweep_threads.py (1 to 4 threads, 3 repeats each) and a combined results table (fp32 vs int8, per thread count, stride 1 vs 2 vs 3). Include a thermal log (temperature sampled every 5 s during a full 60 s run) and report any throttling. Output benchmark_summary.md. STOP.

## STAGE 8: Accuracy, zones, report
- make_ground_truth_template.py: pick 10 to 15 evenly spaced frames, save them as images and create an empty CSV (frame, person, bicycle, car, motorcycle, bus, truck) for me to fill by hand.
- evaluate_accuracy.py: compute MAE per class vs ground truth, before and after smoothing.
- zones.py: polygon ROI counting using box centre; config-driven; draw the zone on annotated frames; add in-zone counts to the CSV.
- report.py: summary.md (peak, average, vehicle-to-pedestrian ratio, congestion label, chart PNG, one rule-based planning insight sentence) and a failure-case export (save 3 frames with the largest count difference vs ground truth, if ground truth exists).
- Privacy: Gaussian-blur person boxes in saved sample frames when enabled.
STOP.

## STAGE 9: Pi deployment and docs
Write README: exact Pi setup steps (OS, Python, venv, installing or building the ExecuTorch runtime for ARM64, with the real commands I need to run), how to copy the .pte, how to start the server, how to run the CLI, how to reproduce benchmarks, known limitations, and a "results" section that stays EMPTY until I paste measured numbers. Add a reproducibility block (OS version, versions, model hash, config used). STOP.

# DEFINITION OF DONE (checklist you must print at the end of Stage 9)
[ ] Upload page works on the Pi from a laptop browser
[ ] 60 s clip processed; processing time reported vs the 60 s target
[ ] CSV, 2+ annotated frames, benchmark.json, summary.md produced
[ ] fp32 vs int8 and thread sweep tables produced from real runs
[ ] Ground-truth MAE reported (or "NOT MEASURED")
[ ] No cloud calls, no tracking, no hard-coded results
[ ] Unit tests pass
