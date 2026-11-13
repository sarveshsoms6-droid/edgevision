"""Export pretrained PyTorch detection model to ExecuTorch FP32 .pte format with XNNPACK.

Follows the MODEL EXPORT DESIGN:
- Exports ONLY the neural network (backbone plus detection heads) as an nn.Module
  taking float tensor [1, 3, H, W] and returning raw class logits and box regression deltas.
- Precomputes anchor boxes and saves them as models/anchors.npy.
- Lowers and partitions the exported graph using ExecuTorch XnnpackPartitioner.
- Prints exact torch and executorch versions.
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
from typing import Tuple

# Ensure flatc.exe is accessible if installed in Python Scripts
_possible_scripts = [
    Path(sys.prefix) / "Scripts",
    Path(os.path.expanduser("~")) / "AppData" / "Local" / "Packages" / "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0" / "LocalCache" / "local-packages" / "Python312" / "Scripts",
]
for s_dir in _possible_scripts:
    flatc_candidate = s_dir / "flatc.exe"
    if flatc_candidate.is_file():
        os.environ["FLATC_EXECUTABLE"] = str(flatc_candidate)
        os.environ["PATH"] = str(s_dir) + os.pathsep + os.environ.get("PATH", "")
        break

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.image_list import ImageList

import executorch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

from edgevision.config import load_config


class SSDLiteExportModule(nn.Module):
    """Clean neural network container without NMS or postprocessing."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        head_outputs = self.head(list(features.values()))
        return head_outputs["cls_logits"], head_outputs["bbox_regression"]


def export_fp32_model(
    config_path: str = "config.yaml",
    output_pte: str | None = None,
    output_anchors: str | None = None,
    weights_path: str | None = None,
) -> Path:
    """Export SSDLite320 to ExecuTorch FP32 .pte format with XNNPACK partitioner."""
    cfg = load_config(config_path)

    torch_ver = torch.__version__
    try:
        et_ver = importlib.metadata.version("executorch")
    except Exception:
        et_ver = "unknown"

    print("=" * 70)
    print("EXECUTORCH FP32 MODEL EXPORT")
    print(f"PyTorch Version    : {torch_ver}")
    print(f"ExecuTorch Version : {et_ver}")
    print(f"Architecture       : {cfg.model.name}")
    print(f"Input Size         : {cfg.model.input_size}x{cfg.model.input_size}")
    if weights_path:
        print(f"Custom Weights     : {weights_path}")
    print("=" * 70)

    # 1. Load pretrained or fine-tuned model
    print("[1/5] Loading detection model weights...")
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    full_model = ssdlite320_mobilenet_v3_large(weights=weights)
    if weights_path and Path(weights_path).is_file():
        print(f"      Loading fine-tuned checkpoint from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu")
        full_model.load_state_dict(state_dict)
    full_model = full_model.eval()

    # 2. Precompute default anchors
    anchors_dest = Path(output_anchors or cfg.model.anchors_path)
    anchors_dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[2/5] Precomputing anchor boxes for {cfg.model.input_size}x{cfg.model.input_size} input...")
    dummy_input = torch.zeros(1, 3, cfg.model.input_size, cfg.model.input_size)
    features = full_model.backbone(dummy_input)
    img_list = ImageList(dummy_input, [(cfg.model.input_size, cfg.model.input_size)])
    anchors = full_model.anchor_generator(img_list, list(features.values()))[0].detach().cpu().numpy()
    np.save(str(anchors_dest), anchors)
    print(f"      Saved anchors: {anchors_dest} (shape {anchors.shape}, {anchors_dest.stat().st_size} bytes)")

    # 3. Build export module
    print("[3/5] Wrapping backbone and detection head into export module...")
    export_module = SSDLiteExportModule(full_model).eval()

    # 4. Export with torch.export
    print("[4/5] Tracing and capturing graph with torch.export.export...")
    sample_input = (dummy_input,)
    exported_program = torch.export.export(export_module, sample_input)

    # 5. Lower to Edge and partition with XNNPACK
    print("[5/5] Lowering to Edge dialect and partitioning with XnnpackPartitioner...")
    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )

    print("      Serializing ExecuTorch program to binary...")
    executorch_program = edge_program.to_executorch()
    pte_buffer = executorch_program.buffer

    pte_dest = Path(output_pte or cfg.model.pte_path)
    pte_dest.parent.mkdir(parents=True, exist_ok=True)
    with open(pte_dest, "wb") as f:
        f.write(pte_buffer)

    file_size_mb = pte_dest.stat().st_size / (1024.0 * 1024.0)
    print("=" * 70)
    print("EXPORT COMPLETED SUCCESSFULLY")
    print(f"Exported Model Path : {pte_dest}")
    print(f"Model File Size     : {file_size_mb:.2f} MB ({pte_dest.stat().st_size:,} bytes)")
    print(f"Anchors Path        : {anchors_dest}")
    print("=" * 70)

    return pte_dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch SSDLite320 to ExecuTorch FP32 .pte")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", type=str, default=None, help="Output .pte destination")
    parser.add_argument("--anchors", type=str, default=None, help="Output anchors.npy destination")
    parser.add_argument("--weights", type=str, default=None, help="Path to custom/fine-tuned .pth weights")
    args = parser.parse_args()

    export_fp32_model(
        config_path=args.config,
        output_pte=args.output,
        output_anchors=args.anchors,
        weights_path=args.weights,
    )


if __name__ == "__main__":
    main()
