"""YOLOv11 detector wrapper for real-time edge and desktop inference.

Integrates Ultralytics YOLOv11 (e.g. yolo11n.pt) for vehicle and pedestrian detection,
supporting both standard COCO pretrained weights and custom vehicle datasets
(e.g., Car, Bus, Ambulance, Motorcycle, Truck).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class YOLOv11Detector:
    """YOLOv11 detector wrapper using Ultralytics engine."""

    def __init__(
        self,
        model_path: str | Path = "models/yolo11n.pt",
        conf_threshold: float = 0.25,
        nms_iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        """Initialize YOLOv11 model.

        Args:
            model_path: Path to YOLO weights (.pt file).
            conf_threshold: Default confidence threshold.
            nms_iou: Default NMS IoU threshold.
            device: Inference device ('cpu', 'cuda', etc.).

        Raises:
            ImportError: If ultralytics is not installed.
            FileNotFoundError: If the model file cannot be found.
        """
        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise ImportError(
                "Ultralytics is required for YOLOv11 inference. Install with: pip install ultralytics"
            ) from err

        p = Path(model_path)
        if not p.is_file():
            candidates = [
                Path("models") / p.name,
                Path("edgevision/models") / p.name,
                Path(__file__).resolve().parent.parent / "models" / p.name,
                Path(p.name),
            ]
            for cand in candidates:
                if cand.is_file():
                    p = cand
                    break

        if not p.is_file():
            raise FileNotFoundError(
                f"YOLO model weights file not found at: {model_path}. "
                "Ensure models/yolo11n.pt exists or specify a valid weights file."
            )

        self.model_path = p
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.device = device
        self.runtime_name = "yolov11"

        # Load YOLO model
        self.model = YOLO(str(self.model_path))

        # Extract class mapping {id: lowercase_name}
        raw_names = getattr(self.model, "names", {})
        if isinstance(raw_names, dict):
            self.id_to_class_map: Dict[int, str] = {
                int(k): str(v).lower().strip() for k, v in raw_names.items()
            }
        elif isinstance(raw_names, (list, tuple)):
            self.id_to_class_map = {
                idx: str(name).lower().strip() for idx, name in enumerate(raw_names)
            }
        else:
            self.id_to_class_map = {}

    def warmup(self, num_iterations: int = 3, input_size: int = 640) -> None:
        """Run dummy inference iterations to warm up PyTorch and Ultralytics caches.

        Args:
            num_iterations: Number of dummy inference cycles.
            input_size: Dimension of dummy input frame.
        """
        dummy = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        for _ in range(max(1, num_iterations)):
            _ = self.model.predict(
                source=dummy,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )

    def predict_detections(
        self,
        frame_bgr: np.ndarray,
        conf_threshold: Optional[float] = None,
        nms_iou: Optional[float] = None,
        max_detections: int = 100,
        target_classes: Optional[Union[List[str], set]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run YOLO inference and return normalized bounding boxes, scores, and labels.

        Args:
            frame_bgr: Input BGR image (H, W, 3).
            conf_threshold: Confidence score cutoff (0.0 to 1.0).
            nms_iou: Non-Maximum Suppression IoU threshold.
            max_detections: Maximum detections to return.
            target_classes: Optional whitelist of class name strings to retain.

        Returns:
            Tuple of:
                boxes: Normalized float32 coordinates [K, 4] in [x1, y1, x2, y2] format in [0.0, 1.0].
                scores: Float32 detection confidence scores [K].
                labels: Int64 category IDs [K].
        """
        conf = float(conf_threshold if conf_threshold is not None else self.conf_threshold)
        iou = float(nms_iou if nms_iou is not None else self.nms_iou)

        results = self.model.predict(
            source=frame_bgr,
            conf=conf,
            iou=iou,
            max_det=max_detections,
            device=self.device,
            verbose=False,
        )

        if not results or len(results) == 0 or results[0].boxes is None:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        boxes_res = results[0].boxes

        if len(boxes_res) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        # Extract normalized boxes [x1, y1, x2, y2] in [0, 1]
        boxes = boxes_res.xyxyn.detach().cpu().numpy().astype(np.float32)
        scores = boxes_res.conf.detach().cpu().numpy().astype(np.float32)
        labels = boxes_res.cls.detach().cpu().numpy().astype(np.int64)

        # Clip boxes to [0.0, 1.0]
        boxes = np.clip(boxes, 0.0, 1.0)

        # Filter by target class whitelist if provided
        if target_classes is not None:
            target_set = {c.lower() for c in target_classes}
            keep_indices = [
                i
                for i, cid in enumerate(labels)
                if self.id_to_class_map.get(int(cid), "") in target_set
            ]
            if len(keep_indices) == 0:
                return (
                    np.empty((0, 4), dtype=np.float32),
                    np.empty(0, dtype=np.float32),
                    np.empty(0, dtype=np.int64),
                )
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

        return boxes, scores, labels
