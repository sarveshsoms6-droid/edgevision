"""PyTorch reference detector wrapper for laptop validation and baseline comparisons.

Wraps torchvision SSDLite320 MobileNetV3 Large with pretrained COCO weights.
Outputs raw predictions or standard boxes for pipeline correctness verification.
"""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)


class PyTorchDetector:
    """Reference PyTorch detector for laptop validation and export verification."""

    def __init__(self, model_name: str = "ssdlite320_mobilenet_v3_large") -> None:
        """Initialize PyTorch detector with pretrained weights."""
        self.model_name = (
            model_name
            if model_name == "ssdlite320_mobilenet_v3_large"
            else "ssdlite320_mobilenet_v3_large"
        )
        self.runtime_name = "pytorch-reference"

        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=weights)
        self.model.eval()

    def predict_raw(
        self, input_data: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference and return raw class logits and box regression deltas.

        Args:
            input_data: Normalized float tensor or numpy array [1, 3, H, W] in range [-1, 1].

        Returns:
            Tuple of (class_logits [N, num_classes], box_regression [N, 4]) as NumPy arrays.
        """
        if isinstance(input_data, np.ndarray):
            tensor = torch.from_numpy(input_data).float()
        else:
            tensor = input_data.float()

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            features = self.model.backbone(tensor)
            head_outputs = self.model.head(list(features.values()))
            cls_logits = head_outputs["cls_logits"][0].cpu().numpy()
            bbox_regression = head_outputs["bbox_regression"][0].cpu().numpy()

        return cls_logits, bbox_regression
