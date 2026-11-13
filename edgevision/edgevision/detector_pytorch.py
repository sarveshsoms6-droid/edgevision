"""PyTorch reference detector wrapper for laptop validation and baseline comparisons.

Supports multiple torchvision COCO detectors via MODEL_REGISTRY:
- SSDLite320 MobileNetV3 Large (SSD family: raw logits output, needs anchor decoding)
- Faster R-CNN MobileNetV3 Large 320 FPN (FRCNN family: built-in RPN+NMS, direct detections)
- Faster R-CNN MobileNetV3 Large FPN (FRCNN family: higher-res variant)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)

# ---------------------------------------------------------------------------
# Model Registry: maps model name -> factory, weights, and family type
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict] = {
    "ssdlite320_mobilenet_v3_large": {
        "factory": ssdlite320_mobilenet_v3_large,
        "weights": SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        "family": "ssd",
        "norm_mode": "ssd",  # [-1, 1] normalization
    },
    "fasterrcnn_mobilenet_v3_large_320_fpn": {
        "factory": fasterrcnn_mobilenet_v3_large_320_fpn,
        "weights": FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1,
        "family": "fasterrcnn",
        "norm_mode": "fasterrcnn",  # [0, 1] normalization
    },
    "fasterrcnn_mobilenet_v3_large_fpn": {
        "factory": fasterrcnn_mobilenet_v3_large_fpn,
        "weights": FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
        "family": "fasterrcnn",
        "norm_mode": "fasterrcnn",  # [0, 1] normalization
    },
}


def get_model_family(model_name: str) -> str:
    """Return the model family ('ssd' or 'fasterrcnn') for a given model name."""
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return entry["family"]


def get_norm_mode(model_name: str) -> str:
    """Return the normalization mode ('ssd' or 'fasterrcnn') for a model."""
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        return "ssd"
    return entry.get("norm_mode", "ssd")


class PyTorchDetector:
    """Reference PyTorch detector for laptop validation and export verification.

    Supports both SSD (raw output) and Faster R-CNN (direct detection) families.
    """

    def __init__(self, model_name: str = "ssdlite320_mobilenet_v3_large") -> None:
        """Initialize PyTorch detector with pretrained weights."""
        self.model_name = model_name
        self.runtime_name = "pytorch-reference"

        entry = MODEL_REGISTRY.get(model_name)
        if entry is None:
            raise ValueError(
                f"Unsupported model architecture: {model_name!r}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        self.family = entry["family"]
        self.norm_mode = entry.get("norm_mode", "ssd")
        self.model = entry["factory"](weights=entry["weights"])
        self.model.eval()

    def predict_raw(
        self, input_data: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference and return raw class logits and box regression deltas.

        Only valid for SSD-family models. Raises RuntimeError for Faster R-CNN.

        Args:
            input_data: Normalized float tensor or numpy array [1, 3, H, W].
                        SSD expects [-1, 1] range.

        Returns:
            Tuple of (class_logits [N, num_classes], box_regression [N, 4]) as NumPy arrays.
        """
        if self.family != "ssd":
            raise RuntimeError(
                f"predict_raw() is only supported for SSD-family models, "
                f"not {self.family!r}. Use predict_detections() instead."
            )

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

    def predict_detections(
        self,
        input_data: Union[np.ndarray, torch.Tensor],
        conf_threshold: float = 0.3,
        target_class_ids: Optional[Dict[int, str]] = None,
        class_thresholds: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference and return filtered detections directly.

        For Faster R-CNN: uses built-in RPN + ROI pooling + NMS.
        For SSD: falls back to predict_raw() (caller must post-process).

        Args:
            input_data: Preprocessed float tensor/array [1, 3, H, W].
                        Faster R-CNN expects [0, 1] range.
            conf_threshold: Minimum confidence for filtering.
            target_class_ids: COCO class ID -> name map for filtering.
            class_thresholds: Optional per-class confidence thresholds.

        Returns:
            Tuple of (boxes [K, 4] normalized 0-1, scores [K], class_ids [K]).
        """
        if target_class_ids is None:
            target_class_ids = {
                1: "person", 2: "bicycle", 3: "car",
                4: "motorcycle", 6: "bus", 8: "truck",
            }

        if isinstance(input_data, np.ndarray):
            tensor = torch.from_numpy(input_data).float()
        else:
            tensor = input_data.float()

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # For Faster R-CNN, the model expects [0, 1] range.
        # If input is in [-1, 1] (SSD normalization), convert it.
        if self.family == "fasterrcnn" and self.norm_mode == "fasterrcnn":
            # Input should already be in [0, 1] if preprocessor is set correctly.
            # But clamp to be safe.
            tensor = tensor.clamp(0.0, 1.0)

        with torch.no_grad():
            if self.family == "fasterrcnn":
                # Faster R-CNN handles its own transforms internally;
                # we pass a list of images (each [C, H, W]) as per torchvision API.
                images = [tensor[0]]
                outputs = self.model(images)
                det = outputs[0]

                raw_boxes = det["boxes"].cpu().numpy()     # [K, 4] pixel coords
                raw_scores = det["scores"].cpu().numpy()   # [K]
                raw_labels = det["labels"].cpu().numpy()    # [K] (COCO IDs)

                # Get image dimensions for normalization
                _, _, h, w = tensor.shape

                # Filter by target classes and confidence thresholds
                keep_mask = np.zeros(len(raw_scores), dtype=bool)
                for i in range(len(raw_scores)):
                    cid = int(raw_labels[i])
                    if cid not in target_class_ids:
                        continue
                    cls_name = target_class_ids[cid]
                    thresh = conf_threshold
                    if class_thresholds and cls_name in class_thresholds:
                        thresh = class_thresholds[cls_name]
                    if raw_scores[i] >= thresh:
                        keep_mask[i] = True

                boxes = raw_boxes[keep_mask]
                scores = raw_scores[keep_mask]
                labels = raw_labels[keep_mask]

                # Normalize boxes to [0, 1]
                if len(boxes) > 0:
                    boxes[:, [0, 2]] /= float(w)
                    boxes[:, [1, 3]] /= float(h)
                    boxes = np.clip(boxes, 0.0, 1.0)

                return (
                    boxes.astype(np.float32),
                    scores.astype(np.float32),
                    labels.astype(np.int64),
                )
            else:
                # SSD: raw output needs external post-processing
                raise RuntimeError(
                    "predict_detections() for SSD requires external post-processing. "
                    "Use predict_raw() + postprocess_detections() instead."
                )

    def warmup(self, num_iterations: int = 5, input_size: int = 320) -> None:
        """Run dummy inferences to warm up the model (excluded from benchmarks)."""
        dummy = torch.zeros(1, 3, input_size, input_size)
        if self.family == "fasterrcnn":
            # Faster R-CNN expects [0, 1] range
            dummy = torch.rand(1, 3, input_size, input_size)
            for _ in range(num_iterations):
                with torch.no_grad():
                    self.model([dummy[0]])
        else:
            # SSD: pass through backbone + head
            for _ in range(num_iterations):
                with torch.no_grad():
                    features = self.model.backbone(dummy)
                    self.model.head(list(features.values()))
