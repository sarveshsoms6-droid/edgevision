# EdgeVision: AI Vehicle & Pedestrian Detection System

**Offline Perception, Traffic Analytics & Real-Time Monitoring Pipeline**  
*Powered by Ultralytics YOLOv11 & ExecuTorch (XNNPACK Edge Backend)*

---

## 📋 Table of Contents

1. [Quick Start (One-Click Launch on Windows)](#-quick-start-one-click-launch-on-windows)
2. [Windows Setup & Installation Guide](#-windows-setup--installation-guide)
   - [Prerequisites](#prerequisites)
   - [Step 1: Open Terminal](#step-1-open-terminal)
   - [Step 2: Navigate to Project Directory](#step-2-navigate-to-project-directory)
   - [Step 3: Create & Activate Virtual Environment](#step-3-create--activate-virtual-environment)
   - [Step 4: Install Dependencies](#step-4-install-dependencies)
   - [Step 5: Verify Installation with Tests](#step-5-verify-installation-with-tests)
3. [Running the Web Application](#-running-the-web-application)
4. [YOLOv11 Workflow: Custom Training & User Testing Folders](#-yolov11-workflow-custom-training--user-testing-folders)
   - [Folder Roles (`training/` vs `testing/`)](#folder-roles-training-vs-testing)
   - [Training YOLOv11 from Custom Images](#training-yolov11-from-custom-images)
   - [Evaluating Model on User Testing Inputs](#evaluating-model-on-user-testing-inputs)
5. [Command-Line Interface (CLI) Usage](#-command-line-interface-cli-usage)
6. [Raspberry Pi 4 Deployment Guide (Debian 12 Bookworm ARM64)](#-raspberry-pi-4-deployment-guide)
7. [Troubleshooting & FAQ](#-troubleshooting--faq)
8. [Project Architecture & Directory Layout](#-project-architecture--directory-layout)

---

## 🚀 Quick Start (One-Click Launch on Windows)

If you are on Windows, you can start the application with a single click:

1. Double-click **`start_server.bat`** (located in the project folder).
2. The server will start, and your web browser will automatically open to:
   ```
   http://127.0.0.1:5000
   ```
3. Drag and drop any traffic video (`.mp4`, `.avi`, `.mov`, `.mkv`) or image (`.png`, `.jpg`), select your detector (`YOLOv11` or `ExecuTorch INT8`), and view real-time vehicle counts, congestion analysis, and annotated outputs.

---

## 💻 Windows Setup & Installation Guide

Follow these exact steps if you are setting up the project from scratch on Windows 10/11.

### Prerequisites

- **Python 3.10 to 3.12** installed on Windows.  
  Verify in your terminal:
  ```powershell
  python --version
  ```
  *(Expected: Python 3.10.x, 3.11.x, or 3.12.x)*
- **Git** (optional, for cloning).

---

### Step 1: Open Terminal

Open **PowerShell** or **Command Prompt (cmd)** on your PC:
- Press `Win + X` and select **Terminal** / **PowerShell** or search for **cmd**.

---

### Step 2: Navigate to Project Directory

Change directory into the `edgevision` folder:

```powershell
cd "c:\Users\Peria Durai\Desktop\raspberry pi\edgevision"
```

> [!TIP]
> If you cloned the project to a different folder, replace the path above with your actual folder path. Always make sure you are inside the folder containing `requirements.txt` and `run.py`.

---

### Step 3: Create & Activate Virtual Environment

A Python virtual environment ensures package dependencies remain clean and isolated.

#### If using Windows PowerShell:

```powershell
# 1. Allow script execution in current PowerShell session (if restricted):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2. Create the virtual environment:
python -m venv .venv

# 3. Activate the virtual environment:
.\.venv\Scripts\Activate.ps1
```

*(You will see `(.venv)` appear at the beginning of your command prompt)*

#### If using Command Prompt (cmd):

```cmd
:: 1. Create the virtual environment:
python -m venv .venv

:: 2. Activate the virtual environment:
.venv\Scripts\activate.bat
```

---

### Step 4: Install Dependencies

Upgrade `pip` and install all required packages:

```powershell
# Upgrade pip:
python -m pip install --upgrade pip

# Install project requirements:
pip install -r requirements.txt
```

#### What `requirements.txt` installs:
- **`ultralytics>=8.4.0`**: YOLOv11 object detection framework.
- **`Flask>=3.0.0` & `Werkzeug>=3.0.0`**: Offline web application server.
- **`opencv-python-headless>=4.10.0`**: Computer vision and video decoding.
- **`torch==2.15.0` & `torchvision==0.29.2`**: PyTorch neural runtime.
- **`executorch==1.6.0`**: Embedded ExecuTorch XNNPACK inference runtime.
- **`numpy`, `psutil`, `matplotlib`, `pyyaml`, `pytest`**: Core utilities and testing.

> [!NOTE]
> If you prefer a quick manual install for inference and web app only, run:
> ```powershell
> pip install ultralytics flask werkzeug opencv-python-headless torch torchvision pyyaml psutil matplotlib pytest
> ```

---

### Step 5: Verify Installation with Tests

Run the automated test suite to confirm everything is working properly:

```powershell
python -m pytest tests -v
```

**Expected Result:**
```
======================= 55 passed, 2 warnings in ~45s =======================
```
All 55 unit tests covering YOLOv11 detection, ExecuTorch INT8 runtime, video I/O, polygon ROI zones, rolling median count smoothing, and web server routes will pass.

---

## 🌐 Running the Web Application

### Method 1: One-Click Batch Script
Double-click `start_server.bat` in the project root or in `edgevision/`.

### Method 2: Command Line
From the `edgevision` folder with your environment activated:

```powershell
python -m app.server
```

### Accessing the Web Interface
- Open your browser to: **`http://127.0.0.1:5000`**
- **Available Models**:
  - **YOLOv11 Nano (`yolo11n.pt`)**: Real-time vehicle classification (Car, Bus, Ambulance, Motorcycle, Truck, Person, Bicycle).
  - **ExecuTorch INT8 (`ssdlite320_mobilenet_v3_large_int8.pte`)**: Ultra-lightweight quantized edge model for Raspberry Pi.
  - **ExecuTorch FP32**: Unquantized edge baseline.
  - **PyTorch Reference**: Standard PyTorch baseline.
- **Features**:
  - Live progress bar polling with frame-by-frame status.
  - Automatic frame skipping (`stride: auto`) for maximum throughput.
  - Single-job concurrency lock (safeguards Pi/CPU RAM from exhaustion).
  - Full analytics dashboard: Peak vehicle counts, per-class breakdown, congestion level classification (`Low`, `Medium`, `High`), and downloadable annotated video clips and CSV reports.

---

## 🚗 YOLOv11 Workflow: Custom Training & User Testing Folders

EdgeVision includes a streamlined two-folder workflow specifically tailored for custom model training and continuous user testing:

```
edgevision/
├── training/   <-- STRICTLY EMPTY (Paste your .png training images here)
└── testing/    <-- USER TESTING (Stores every uploaded video/image automatically)
```

*(Note: In the root repository, `training` and `testing` are linked directly to `edgevision/training` and `edgevision/testing`)*

### Folder Roles (`training/` vs `testing/`)

| Folder | Status | Purpose |
|---|---|---|
| **`training/`** | **Strictly Empty** | Reserved exclusively for user training data. Paste your custom `.png` images and annotation files into this folder. |
| **`testing/`** | **User Inputs** | Automatically captures every video or image submitted via the web interface or CLI. Used to test and evaluate the model. |

---

### Training YOLOv11 from Custom Images

1. Copy your `.png` training images into the `training/` folder:
   ```
   training/
   ├── image1.png
   ├── image2.png
   └── ...
   ```
2. Run the automated training script:
   ```powershell
   python scripts/train_from_folder.py --epochs 50 --batch 8 --imgsz 640
   ```
3. The script automatically creates the YOLO dataset configuration, validates the images, fine-tunes `yolo11n.pt`, and exports the trained weights into `models/yolo11_custom.pt`.

---

### Evaluating Model on User Testing Inputs

Every time you upload a video or image via the web app (or pass an input via `run.py`), a copy is automatically archived into `testing/`.

To run an offline batch evaluation across all user inputs stored in `testing/`:

```powershell
python scripts/test_from_folder.py --conf 0.40
```

**Output Generated**:
- Frame-by-frame detection counts for each test video/image.
- Annotated output videos and images saved under `outputs/testing_results/`.
- Per-input CSV count summaries.

---

## ⌨️ Command-Line Interface (CLI) Usage

EdgeVision can also be run entirely headless from the terminal using `run.py`.

### 1. Run with YOLOv11 Detector (Recommended)
```powershell
python run.py --input sample_data/traffic_sample.mp4 --runtime yolov11
```

### 2. Run with ExecuTorch Quantized INT8
```powershell
python run.py --input sample_data/traffic_sample.mp4 --runtime executorch-int8 --stride auto
```

### 3. Run with ExecuTorch FP32
```powershell
python run.py --input sample_data/traffic_sample.mp4 --runtime executorch-fp32 --stride 2
```

### 4. Enable ROI Zone Counting
Count vehicles entering a specific polygon Region of Interest:
```powershell
python run.py --input sample_data/traffic_sample.mp4 --runtime yolov11 --zone
```

### CLI Arguments Reference

| Flag | Default | Description |
|---|---|---|
| `--input` | *required* | Path to input traffic video (`.mp4`, `.avi`, `.mov`) or image (`.png`, `.jpg`). |
| `--runtime` | `yolov11` | Inference runtime: `yolov11`, `executorch-int8`, `executorch-fp32`, `pytorch-reference`. |
| `--stride` | `auto` | Frame decoding stride: `auto` (adaptive dynamic stride) or fixed integer (`1`, `2`, `5`). |
| `--zone` | `False` | Enable polygon Region of Interest (ROI) zone filtering. |
| `--config` | `config.yaml` | Path to pipeline YAML configuration file. |
| `--output-dir`| `outputs/job_<uuid>` | Directory where artifacts, CSV, and sample frames are saved. |

---

## 🍓 Raspberry Pi 4 Deployment Guide

EdgeVision is engineered to run offline on a **Raspberry Pi 4 Model B (4 GB / 8 GB RAM)** running **64-bit Raspberry Pi OS (Debian 12 Bookworm, aarch64)**.

### 1. Hardware & OS Check
On your Raspberry Pi terminal, verify you are running 64-bit:
```bash
uname -m
# Output MUST be: aarch64
```

### 2. Install System Packages
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1-mesa-glx libglib2.0-0 libraspberrypi-bin
```

### 3. Setup Python Virtual Environment
```bash
cd ~
python3 -m venv edgevision-venv
source edgevision-venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 4. Transfer Files to the Pi
From your Windows development PC (using PowerShell):
```powershell
# Using scp:
scp -r "c:\Users\Peria Durai\Desktop\raspberry pi\edgevision" pi@<PI_IP_ADDRESS>:~/edgevision
```

### 5. Install Dependencies on Pi
```bash
cd ~/edgevision
pip install -r requirements.txt
```

### 6. Run on Raspberry Pi
```bash
# Launch the web interface (accessible from any device on your Wi-Fi):
python -m app.server

# Or run via CLI with ExecuTorch INT8 (optimized for ARM64 CPU):
python run.py --input sample_data/traffic_sample.mp4 --runtime executorch-int8 --stride auto
```

Access the web interface at `http://<YOUR_PI_IP>:5000` from your laptop or phone on the same network.

---

## 🛠️ Troubleshooting & FAQ

### Q1: `File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system` (PowerShell)
**Solution**: Windows restricts script execution by default. Run this in your PowerShell window:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then activate again:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Q2: `'sudo' is not recognized as an internal or external command`
**Cause**: `sudo` is a Linux command. You are on Windows!  
**Solution**: Follow the [Windows Setup & Installation Guide](#-windows-setup--installation-guide) section above. Do not run `sudo apt-get` on Windows.

### Q3: `'source' is not recognized as an internal or external command`
**Cause**: `source` is a Linux bash shell command.  
**Solution**: On Windows PowerShell, activate using `.\.venv\Scripts\Activate.ps1`. On Command Prompt (cmd), use `.venv\Scripts\activate.bat`.

### Q4: `Error: [Errno 22] Invalid argument`
**Cause**: Previously occurred in OpenCV `VideoWriter` when decoding non-standard resolution frames or odd dimensions.  
**Solution**: Fully resolved! The pipeline now validates frame dimensions before initializing `VideoWriter`, ensures matching `(width, height)` tuples, and falls back gracefully to `MJPG` or raw frame extraction if the host codec is unavailable.

### Q5: Port 5000 already in use
**Solution**: If another process is holding port 5000, find and terminate it:
```powershell
# Find process using port 5000:
netstat -ano | findstr :5000

# Kill process by PID (replace 1234 with the PID from previous command):
taskkill /F /PID 1234
```
Or start the server on a different port:
```powershell
python -m app.server --port 5050
```

---

## 📁 Project Architecture & Directory Layout

```
edgevision/
├── start_server.bat            # One-click Windows launcher for web server
├── run.py                      # CLI inference and pipeline entry point
├── config.yaml                 # User-configurable detection & threshold parameters
├── requirements.txt            # Pinned dependencies
├── training/                   # STRICTLY EMPTY: User pastes .png training images here
├── testing/                    # USER INPUTS: Automatically stores all user uploads
│
├── app/                        # Offline Flask Web Application
│   ├── server.py               # Web server & single-job background thread manager
│   ├── static/style.css        # Minimalist offline styles (zero external CDNs)
│   └── templates/              # HTML templates (index, status, result)
│
├── edgevision/                 # Perception & Analytics Core Library
│   ├── pipeline.py             # Orchestrates decoding -> detection -> smoothing -> export
│   ├── detector_yolo.py        # YOLOv11 multi-class vehicle detector
│   ├── detector_et.py          # ExecuTorch XNNPACK INT8/FP32 runtime wrapper
│   ├── detector_pytorch.py     # PyTorch reference detector
│   ├── video_io.py             # Bounded-queue producer-consumer frame decoder
│   ├── postprocess.py          # Vectorised NumPy NMS and bounding-box regression
│   ├── counter.py              # Per-class detection aggregator
│   ├── smoothing.py            # Rolling median temporal filter (eliminates jitter)
│   ├── zones.py                # Ray-casting point-in-polygon ROI counting
│   ├── annotate.py             # Visual bounding boxes & privacy blur
│   ├── benchmark.py            # Latency percentiles, RSS RAM, CPU temp metrics
│   └── report.py               # Summary markdown, trend charts, congestion analysis
│
├── models/                     # Deep learning model artifacts
│   ├── yolo11n.pt              # YOLOv11 nano model weights (5.35 MB)
│   ├── ssdlite320_mobilenet_v3_large_int8.pte # Quantized ExecuTorch INT8 model (3.68 MB)
│   ├── ssdlite320_mobilenet_v3_large.pte      # FP32 ExecuTorch model (13.15 MB)
│   └── anchors.npy             # Precomputed anchor boxes for SSD Lite
│
├── scripts/                    # Utility scripts
│   ├── train_from_folder.py    # Fine-tunes YOLOv11 from training/ .png images
│   ├── test_from_folder.py     # Evaluates model across all testing/ inputs
│   ├── export_model.py         # Exports PyTorch model to ExecuTorch .pte
│   ├── quantize_model.py       # PT2E INT8 calibration and quantization
│   └── sweep_threads.py        # Thread scaling benchmark tool
│
├── tests/                      # 55 automated unit & integration tests
│   ├── test_benchmark.py
│   ├── test_counter.py
│   ├── test_detector_yolo.py
│   ├── test_postprocess.py
│   ├── test_quantization.py
│   ├── test_report.py
│   ├── test_server.py
│   ├── test_smoke.py
│   ├── test_smoothing.py
│   └── test_zones.py
│
└── sample_data/                # Sample video clip for immediate testing
    └── traffic_sample.mp4
```

---

## 🧪 Verified Performance & Test Suite

All functions have been verified with automated unit tests on Windows 10 Pro (Python 3.12.10):

| Test Suite | Tests | Result |
|---|:---:|:---:|
| YOLOv11 Detection & Filter Tests (`test_detector_yolo.py`) | 6 | **PASS** |
| ExecuTorch INT8 Runtime & Quantization (`test_quantization.py`) | 1 | **PASS** |
| Post-Processing & Vectorized NMS (`test_postprocess.py`) | 6 | **PASS** |
| Temporal Rolling Median Smoothing (`test_smoothing.py`) | 4 | **PASS** |
| Polygon ROI Zone Ray-Casting (`test_zones.py`) | 4 | **PASS** |
| Offline Web Server & Progress Polling (`test_server.py`) | 7 | **PASS** |
| System Benchmarking & Latency Percentiles (`test_benchmark.py`) | 4 | **PASS** |
| Smoke Tests & Module Imports (`test_smoke.py`) | 17 | **PASS** |
| Per-Class Aggregation & Vehicle Counter (`test_counter.py`) | 3 | **PASS** |
| Report Generation & Chart Plotting (`test_report.py`) | 3 | **PASS** |
| **Total Test Suite** | **55** | **100% PASS** |

To run the complete verification suite at any time:
```powershell
python -m pytest tests -v
```
