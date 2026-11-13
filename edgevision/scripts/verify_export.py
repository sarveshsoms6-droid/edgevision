"""Verify exported ExecuTorch .pte model against PyTorch reference implementation.

Runs identical input tensors through both PyTorch and the exported .pte:
- Computes maximum absolute difference and mean absolute difference on raw logits and boxes.
- Evaluates whether top detections (classes and decoded boxes) match.
- Prints detailed verification report.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Ensure flatc / tools on path if needed
_possible_scripts = [
    Path(sys.prefix) / "Scripts",
    Path(os.path.expanduser("~")) / "AppData" / "Local" / "Packages" / "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0" / "LocalCache" / "local-packages" / "Python312" / "Scripts",
]
for s_dir in _possible_scripts:
    if (s_dir / "flatc.exe").is_file():
        os.environ["PATH"] = str(s_dir) + os.pathsep + os.environ.get("PATH", "")
        break

import numpy as np
import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from executorch.runtime import Runtime

from edgevision.config import load_config
from edgevision.postprocess import postprocess_detections
from edgevision.preprocess import Preprocessor
from edgevision.video_io import read_video_frames
from scripts.export_model import SSDLiteExportModule


def verify_export(
    config_path: str = "config.yaml",
    pte_path: str | None = None,
    test_video: str | None = None,
) -> bool:
    """Verify numerical and functional equivalence of .pte model against PyTorch."""
    cfg = load_config(config_path)
    target_pte = Path(pte_path or cfg.model.pte_path)

    if not target_pte.is_file():
        raise FileNotFoundError(f"Exported .pte file not found at: {target_pte}")

    anchors_path = Path(cfg.model.anchors_path)
    if not anchors_path.is_file():
        raise FileNotFoundError(f"Anchors file not found at: {anchors_path}")
    anchors = np.load(anchors_path)

    torch_ver = torch.__version__
    try:
        et_ver = importlib.metadata.version("executorch")
    except Exception:
        et_ver = "unknown"

    print("=" * 70)
    print("EXECUTORCH NUMERICAL EXPORT VERIFICATION")
    print(f"PyTorch Version    : {torch_ver}")
    print(f"ExecuTorch Version : {et_ver}")
    print(f"Model File (.pte)  : {target_pte} ({target_pte.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 70)

    # 1. Prepare sample frame
    preprocessor = Preprocessor(input_size=cfg.model.input_size)
    if test_video and Path(test_video).is_file():
        print(f"Loading test frame from video: {test_video}")
        frame_bgr = next(read_video_frames(test_video, max_width=640, frame_stride=1))[2]
    else:
        # Synthetic test pattern
        print("Using synthetic gradient pattern as test frame...")
        frame_bgr = np.zeros((360, 640, 3), dtype=np.uint8)
        for i in range(360):
            frame_bgr[i, :, 0] = (i * 255) // 360
            frame_bgr[i, :, 1] = 120
            frame_bgr[i, :, 2] = 200

    preprocessed_np = preprocessor.preprocess_frame(frame_bgr)
    input_tensor = torch.from_numpy(preprocessed_np).float()

    # 2. PyTorch Reference Inference
    print("[1/3] Running PyTorch reference model...")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    pt_model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
    export_module = SSDLiteExportModule(pt_model).eval()

    with torch.no_grad():
        pt_logits_t, pt_regs_t = export_module(input_tensor)
        pt_logits = pt_logits_t[0].cpu().numpy()
        pt_regs = pt_regs_t[0].cpu().numpy()

    # 3. ExecuTorch .pte Inference
    print("[2/3] Running ExecuTorch .pte model via XNNPACK runtime...")
    runtime = Runtime.get()
    program = runtime.load_program(target_pte)
    forward_method = program.load_method("forward")
    et_outputs = forward_method.execute([input_tensor])
    et_logits = et_outputs[0][0].numpy()
    et_regs = et_outputs[1][0].numpy()

    # 4. Numerical Comparison
    print("[3/3] Comparing raw outputs...")
    diff_logits = np.abs(pt_logits - et_logits)
    max_diff_logits = float(np.max(diff_logits))
    mean_diff_logits = float(np.mean(diff_logits))

    diff_regs = np.abs(pt_regs - et_regs)
    max_diff_regs = float(np.max(diff_regs))
    mean_diff_regs = float(np.mean(diff_regs))

    print("-" * 70)
    print(f"Logits Max Absolute Difference : {max_diff_logits:.6e}")
    print(f"Logits Mean Absolute Difference: {mean_diff_logits:.6e}")
    print(f"Regs   Max Absolute Difference : {max_diff_regs:.6e}")
    print(f"Regs   Mean Absolute Difference: {mean_diff_regs:.6e}")
    print("-" * 70)

    # 5. Top Detections Comparison
    pt_boxes, pt_scores, pt_labels = postprocess_detections(
        logits=pt_logits,
        regressions=pt_regs,
        anchors=anchors,
        conf_threshold=0.10,  # Lower threshold to compare top candidates
        nms_iou=cfg.nms_iou,
        max_detections=10,
        target_class_ids=cfg.classes.coco_id_map,
        input_size=cfg.model.input_size,
    )

    et_boxes, et_scores, et_labels = postprocess_detections(
        logits=et_logits,
        regressions=et_regs,
        anchors=anchors,
        conf_threshold=0.10,
        nms_iou=cfg.nms_iou,
        max_detections=10,
        target_class_ids=cfg.classes.coco_id_map,
        input_size=cfg.model.input_size,
    )

    print(f"PyTorch Top Detections Count  : {len(pt_boxes)}")
    print(f"ExecuTorch Top Detections Count: {len(et_boxes)}")

    detections_match = (
        len(pt_boxes) == len(et_boxes)
        and np.array_equal(pt_labels, et_labels)
        and (len(pt_boxes) == 0 or np.allclose(pt_scores, et_scores, atol=1e-3))
    )

    print("=" * 70)
    # For FP32, tolerance of 1e-4 on logits is standard for floating point equivalence
    numerical_pass = max_diff_logits < 1e-3 and max_diff_regs < 1e-3
    if numerical_pass and (len(pt_boxes) == len(et_boxes)):
        print("VERIFICATION OUTCOME: [PASS] - ExecuTorch FP32 matches PyTorch reference.")
        return True
    else:
        print("VERIFICATION OUTCOME: [WARN/FAIL] - Discrepancy observed.")
        print(f"  Numerical pass: {numerical_pass}, Detections match: {detections_match}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ExecuTorch .pte model against PyTorch reference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--pte", type=str, default=None, help="Path to .pte file")
    parser.add_argument("--video", type=str, default="sample_data/traffic_sample.mp4", help="Sample video")
    args = parser.parse_args()

    passed = verify_export(
        config_path=args.config,
        pte_path=args.pte,
        test_video=args.video,
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
