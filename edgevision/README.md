# EdgeVision

**Offline Vehicle and Pedestrian Detection and Counting System for Raspberry Pi 4**  
*Powered by ExecuTorch (XNNPACK backend), featuring an offline web upload interface and high-performance CLI.*

---

## Overview

EdgeVision is an embedded-ML perception pipeline engineered specifically for resource-constrained edge computing on the **Raspberry Pi 4 (4 GB, 64-bit Raspberry Pi OS)**. It performs offline detection, classification, counting, and congestion analysis on traffic video clips (up to 60 seconds, 360p–720p).

> [!IMPORTANT]
> **Model Architecture Confirmation:**
> The default detector is torchvision `ssdlite320_mobilenet_v3_large` with pretrained COCO weights. The model is modular and swappable via `config.yaml`. Note that the exact model architecture must be confirmed against the organizers' approved model list prior to final deployment.

---

## Key Technical Highlights

1. **100% Offline & Autonomous:** Zero cloud dependencies, zero external network calls, zero tracking databases. Operates fully on edge hardware.
2. **ExecuTorch XNNPACK Runtime:** Quantized INT8 and FP32 execution lowered directly to Edge IR with multi-threaded XNNPACK CPU acceleration.
3. **Optimized Latency Strategy:**
   - **Bounded Queue Decoder Pipeline:** Two-thread producer-consumer architecture (`VideoDecoderThread`, bounded queue size 4) decoupling video frame decoding from neural inference.
   - **Adaptive Frame Skipping:** Dynamic stride calculation (`auto` mode) ensuring end-to-end processing maintains pace with video playback framerates.
   - **Vectorised NumPy Post-Processing:** Anchor box regression decoding, thresholding, and Non-Maximum Suppression (NMS) executed without Python per-box loops.
   - **Preallocated Buffers & Ingest Downscaling:** Preallocated tensor buffers; high-resolution videos dynamically downscaled to 640px max width at ingestion.
   - **Warm-Up Isolation:** 5 dummy inferences execute prior to timing to exclude JIT/cache initialization latency.
4. **Multi-Modal Analytics & Reporting:**
   - Per-frame counts for target classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
   - Polygon Region of Interest (ROI) zone counting using ray casting.
   - Temporal count smoothing via rolling median filters to eliminate transient detection jitter.
   - Automated congestion classification (`Low`, `Medium`, `High`) and rule-based planning recommendations.
   - Headless trend charts (`counts_chart.png`) and comprehensive summary reports (`summary.md`).
5. **Privacy by Design:** Automatic Gaussian blurring applied over person bounding boxes in exported sample frames.
6. **Minimalist Offline Web Application:** Flask server bound to LAN `0.0.0.0:5000` with client-side polling, real-time progress bar, single-job concurrency locking, and zero CDN/external JS dependencies.

---

## Directory Structure

```
edgevision/
├── PLAN.md                     # Complete project prompt, requirements, and rules
├── README.md                   # System documentation and deployment manual
├── requirements.txt            # Pinned dependencies
├── config.yaml                 # User-configurable pipeline settings
├── run.py                      # CLI processing entry point
├── benchmark_summary.md        # Thread and runtime benchmark sweep results
├── edgevision/                 # Core Python package
│   ├── __init__.py
│   ├── config.py               # YAML configuration loader and dataclass validator
│   ├── video_io.py             # Video ingestion and bounded-queue decoder thread
│   ├── preprocess.py           # Frame preprocessing and buffer recycling
│   ├── detector_pytorch.py     # PyTorch reference detector wrapper (baseline)
│   ├── detector_et.py          # ExecuTorch XNNPACK runtime wrapper
│   ├── postprocess.py          # Vectorised NumPy NMS and anchor decoding
│   ├── counter.py              # Per-class counts and vehicle aggregation
│   ├── zones.py                # Polygon ROI point-in-polygon counting
│   ├── smoothing.py            # Rolling median temporal count smoothing
│   ├── annotate.py             # Visual bounding boxes, labels, and privacy blur
│   ├── pipeline.py             # Orchestrates end-to-end processing pipeline
│   ├── benchmark.py            # Latency percentiles, RSS RAM, CPU temp
│   └── report.py               # summary.md, charts, congestion classification
├── scripts/                    # Utilities and export tools
│   ├── export_model.py         # FP32 .pte export and lowering
│   ├── quantize_model.py       # INT8 .pte post-training quantization (PT2E)
│   ├── verify_export.py        # Numerical parity verification vs PyTorch
│   ├── sweep_threads.py        # Thread scaling benchmark (1-4 threads)
│   ├── make_ground_truth_template.py # Ground truth CSV generator
│   └── evaluate_accuracy.py    # MAE evaluation against ground truth
├── app/                        # Minimal offline web interface
│   ├── server.py               # Flask upload server (LAN 0.0.0.0:5000)
│   ├── static/style.css        # Minimalist offline styles
│   └── templates/
│       ├── index.html          # Upload form
│       ├── status.html         # Real-time progress bar
│       └── result.html         # Analysis dashboard
├── models/                     # .pte models and precomputed anchors.npy
│   ├── anchors.npy             # 3,234 precomputed anchor boxes
│   ├── ssdlite320_mobilenet_v3_large.pte      # FP32 ExecuTorch model (13.15 MB)
│   └── ssdlite320_mobilenet_v3_large_int8.pte # INT8 ExecuTorch model (3.68 MB)
├── ground_truth/               # Human evaluation ground truth
│   ├── ground_truth_template.csv
│   └── samples/                # Extracted frames for manual inspection
├── sample_data/                # Synthetic test clip
├── uploads/                    # Incoming video storage for web server
├── outputs/                    # Processed results per job ID
├── docs/                       # Project logs and final report
│   ├── BUILD_LOG.md            # Stage-by-stage implementation and telemetry log
│   ├── EXPORT_ISSUES.md        # Technical resolution of export hurdles
│   └── FINAL_REPORT.md         # Final delivery report and DoD checklist
└── tests/                      # Automated unit test suite
    ├── test_smoke.py
    ├── test_postprocess.py
    ├── test_zones.py
    ├── test_smoothing.py
    ├── test_counter.py
    ├── test_benchmark.py
    ├── test_quantization.py
    ├── test_report.py
    └── test_server.py
```

---

## Raspberry Pi 4 Deployment Guide

Follow these exact steps to set up and run EdgeVision on a Raspberry Pi 4.

### 1. Hardware & OS Requirements
- **Board**: Raspberry Pi 4 Model B (4 GB or 8 GB RAM recommended).
- **Cooling**: Passive aluminum heatsink case or active cooling fan strongly recommended to prevent thermal throttling under continuous inference.
- **Operating System**: **Raspberry Pi OS 64-bit (Debian 12 Bookworm, aarch64)**.  
  *Verify architecture on the Pi terminal:*
  ```bash
  uname -m
  # Expected output: aarch64
  ```

### 2. System Dependencies
Install system libraries for OpenCV, video decoding, and hardware temperature telemetry:
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1-mesa-glx libglib2.0-0 libraspberrypi-bin
```

### 3. Python Virtual Environment Setup
```bash
cd ~
python3 -m venv edgevision-venv
source edgevision-venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 4. Installing the ExecuTorch ARM64 Runtime
On ARM64 Linux, install ExecuTorch and PyTorch:
```bash
# Option A: Install via PyTorch wheel repository
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
pip install executorch==1.6.0

# Option B: If building ExecuTorch from source for ARM64 with XNNPACK:
git clone --depth=1 --branch v1.6.0 https://github.com/pytorch/executorch.git
cd executorch
git submodule sync && git submodule update --init --recursive
./install_requirements.sh
mkdir cmake-out && cd cmake-out
cmake .. -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_EXTENSION_RUNNER=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
cmake --build . -j4
pip install .
```

### 5. Transferring Project Files to the Pi
From your development workstation, copy the repository (including pre-exported `.pte` models) to the Pi:
```bash
# From laptop / workstation terminal:
rsync -avz --exclude 'uploads/*' --exclude 'outputs/*' edgevision/ pi@<pi-ip>:~/edgevision/
```

### 6. Install Project Dependencies
On the Raspberry Pi:
```bash
cd ~/edgevision
pip install -r requirements.txt
```

---

## Operating Instructions

### Running the Offline Web Interface
Start the Flask web server:
```bash
python -m app.server
```
- Open `http://<pi-ip>:5000` from any browser on the same local network (Wi-Fi or Ethernet).
- **Upload constraints**: Video duration <= 60 seconds, file size <= 100 MB.
- **Single-Job Protection**: The server locks concurrent processing to protect the Pi's 4 GB RAM. Any secondary upload attempt receives a HTTP 429 "System Busy" notice until the active job completes.

### Running via Command Line (CLI)
```bash
# Process using ExecuTorch INT8 (recommended for Pi 4)
python run.py --input sample_data/traffic_sample.mp4 --runtime executorch-int8 --stride auto

# Process with explicit stride and polygon ROI zone counting enabled
python run.py --input sample_data/traffic_sample.mp4 --runtime executorch-int8 --stride 2 --zone

# Process using PyTorch reference (for workstation parity checks)
python run.py --input sample_data/traffic_sample.mp4 --runtime pytorch-reference --stride 2
```

### Exporting and Quantizing Models
To re-export or quantize the neural network:
```bash
# 1. Export FP32 ExecuTorch model (.pte) and precompute anchors
python scripts/export_model.py

# 2. Verify numerical parity between PyTorch and ExecuTorch FP32
python scripts/verify_export.py

# 3. Quantize to INT8 using PT2E and XNNPACKQuantizer with calibration
python scripts/quantize_model.py --frames 100
```

### Reproducing Benchmark Sweeps
Run the thread scaling and runtime benchmark sweep:
```bash
python scripts/sweep_threads.py --repeats 3 --strides 1 2 3 --threads 1 2 4
# Outputs benchmark_summary.md with thermal logging and throttling analysis
```

### Evaluating Ground-Truth Accuracy
```bash
# 1. Generate 12 sample frames and unpopulated CSV template
python scripts/make_ground_truth_template.py --video sample_data/traffic_sample.mp4 --samples 12

# 2. Inspect frames in ground_truth/samples/ and enter counts in ground_truth/ground_truth.csv

# 3. Evaluate MAE before and after temporal smoothing
python scripts/evaluate_accuracy.py --pred-csv outputs/job_xxxx/counts.csv --gt-csv ground_truth/ground_truth.csv
```

---

## Measured Performance Results (Raspberry Pi 4)

> [!NOTE]
> **Hard Rule on Integrity:** Per project guidelines, benchmark figures must never be fabricated. The table below remains **EMPTY** until real measurements are executed on physical Raspberry Pi 4 hardware. Host workstation figures are documented in `benchmark_summary.md`.

| Runtime | Threads | Stride | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) | Effective FPS | Peak RSS (MB) | Temp Start (°C) | Temp End (°C) |
|---|---|---|---|---|---|---|---|---|---|
| *executorch-int8* | 4 | 2 | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* |
| *executorch-int8* | 4 | auto | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* |
| *executorch-fp32* | 4 | 2 | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* | *NOT MEASURED ON PI* |

---

## Known Limitations

1. **No Object Tracking IDs:** By design, EdgeVision does not implement visual multi-object tracking (DeepSORT, ByteTrack) to maintain edge feasibility on Pi 4 without GPU acceleration. Counts reflect instantaneous frame detections smoothed with rolling median filters.
2. **Thermal Dissipation:** Continuous heavy inference without active cooling may cause the Broadcom BCM2711 SoC to throttle frequency from 1.5 GHz down to 1.0 GHz when reaching 80°C.
3. **Ingest Downscaling:** Input video frames wider than 640px are downscaled during decode using OpenCV to minimize memory bandwidth and decode overhead.
4. **Offline Assumption:** No user authentication or encryption is enabled on the web interface; it is intended solely for local private area network (LAN) operations.

---

## Reproducibility Block

- **Host Development OS**: Windows 10 Pro (10.0.19045 AMD64)
- **Target Embedded OS**: Raspberry Pi OS 64-bit (Debian 12 Bookworm, aarch64, Linux kernel 6.6+)
- **Python Version**: `3.12.10`
- **PyTorch Version**: `2.15.0+cpu`
- **TorchVision Version**: `0.29.2+cpu`
- **ExecuTorch Version**: `1.6.0`
- **Model Architecture**: `torchvision.models.detection.ssdlite320_mobilenet_v3_large` (COCO weights)
- **Model SHA256 Checksums**:
  - `ssdlite320_mobilenet_v3_large.pte` (FP32): `f96167a8b621e8b6e3cae4f9925e97b406d41b9d65e046230084ed112978a822`
  - `ssdlite320_mobilenet_v3_large_int8.pte` (INT8): `6225711b5a4600b5ea30c3ffd0b016a04cac98d00c17779bf04881814a11529f`
  - `anchors.npy`: `bfea495927def833e6902b8ad06099bcc1b88546ff2db7a5494146cc62b55c45`
- **Active Configuration**: `config.yaml`
