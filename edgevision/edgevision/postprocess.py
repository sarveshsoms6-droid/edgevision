"""Vectorised NumPy post-processing for object detection.

Implements:
- Anchor box decoding from box regression deltas.
- Numerical softmax score computations.
- Confidence thresholding and target class filtering BEFORE NMS.
- Vectorised Non-Maximum Suppression (NMS) without Python per-box loops.
- Inter-class vehicle deduplication NMS to eliminate multi-class double-counting
  (e.g., preventing a single vehicle from being detected as both car and truck).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def decode_boxes(
    regression_deltas: np.ndarray,
    anchors: np.ndarray,
    weights: Tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
    bbox_xform_clip: float = math.log(1000.0 / 16.0),
) -> np.ndarray:
    """Decode raw box regression deltas against precomputed anchors into [x1, y1, x2, y2].

    Args:
        regression_deltas: Box regression deltas [N, 4].
        anchors: Precomputed anchor boxes [N, 4] in [x1, y1, x2, y2] format.
        weights: Variance scaling weights (wx, wy, ww, wh), default (10.0, 10.0, 5.0, 5.0).
        bbox_xform_clip: Maximum exponent clip to prevent overflow.

    Returns:
        Decoded bounding boxes [N, 4] in same coordinate scale as anchors.
    """
    if len(regression_deltas) == 0:
        return np.empty((0, 4), dtype=np.float32)

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = regression_deltas[:, 0] / wx
    dy = regression_deltas[:, 1] / wy
    dw = np.clip(regression_deltas[:, 2] / ww, a_min=None, a_max=bbox_xform_clip)
    dh = np.clip(regression_deltas[:, 3] / wh, a_min=None, a_max=bbox_xform_clip)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def nms_vectorized(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.35,
    max_detections: int = 100,
) -> np.ndarray:
    """Perform vectorised Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes: Bounding boxes [N, 4] in [x1, y1, x2, y2] format.
        scores: Confidence scores [N].
        iou_threshold: IoU overlap threshold.
        max_detections: Maximum detections to retain.

    Returns:
        Indices of retained boxes as integer numpy array.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if len(keep) >= max_detections:
            break

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def deduplicate_cross_class_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    vehicle_ids: Sequence[int] = (2, 3, 4, 6, 8),
    iou_threshold: float = 0.40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Suppress overlapping detections across mutually exclusive categories (e.g. car vs truck).

    Whenever two detections within the vehicle group overlap with IoU >= iou_threshold,
    the lower-confidence detection is suppressed to eliminate multi-class double-counting.

    Args:
        boxes: Normalized bounding boxes [K, 4] in range [0.0, 1.0].
        scores: Detection confidence scores [K].
        labels: Detected class IDs [K].
        vehicle_ids: Set of numeric class IDs belonging to vehicles.
        iou_threshold: IoU overlap threshold for suppression.

    Returns:
        Filtered tuple of (boxes, scores, labels).
    """
    if len(boxes) <= 1:
        return boxes, scores, labels

    order = scores.argsort()[::-1]
    keep_indices: List[int] = []
    suppressed = set()

    v_set = set(vehicle_ids)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    for idx in range(len(order)):
        i = int(order[idx])
        if i in suppressed:
            continue
        keep_indices.append(i)
        is_veh_i = int(labels[i]) in v_set

        for jdx in range(idx + 1, len(order)):
            j = int(order[jdx])
            if j in suppressed:
                continue

            is_veh_j = int(labels[j]) in v_set

            # Suppress if both are vehicles or if both share identical label
            if (is_veh_i and is_veh_j) or (labels[i] == labels[j]):
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])

                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                union = areas[i] + areas[j] - inter
                iou = inter / union if union > 0 else 0.0

                if iou >= iou_threshold:
                    suppressed.add(j)

    keep_arr = np.array(sorted(keep_indices), dtype=np.int64)
    return boxes[keep_arr], scores[keep_arr], labels[keep_arr]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def postprocess_detections(
    logits: np.ndarray,
    regressions: np.ndarray,
    anchors: np.ndarray,
    conf_threshold: float = 0.20,
    nms_iou: float = 0.35,
    max_detections: int = 100,
    target_class_ids: Optional[Dict[int, str]] = None,
    input_size: int = 320,
    inter_class_vehicle_nms: bool = True,
    inter_class_iou: float = 0.40,
    class_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute end-to-end postprocessing: softmax, filtering, decoding, and NMS.

    Args:
        logits: Raw class logits [N, num_classes].
        regressions: Raw box regression deltas [N, 4].
        anchors: Precomputed anchors [N, 4].
        conf_threshold: Fallback minimum confidence threshold applied before NMS.
        nms_iou: Intra-class IoU threshold for NMS.
        max_detections: Maximum total detections to keep.
        target_class_ids: Mapping of COCO category ID to class name.
        input_size: Input image dimension used for coordinate normalization.
        inter_class_vehicle_nms: Whether to eliminate overlapping cross-class duplicates.
        inter_class_iou: IoU threshold for cross-class vehicle duplicate suppression.
        class_thresholds: Optional per-class confidence thresholds dictionary.

    Returns:
        Tuple of:
            - boxes [K, 4] normalized in range [0.0, 1.0] [x1, y1, x2, y2]
            - scores [K] confidence values
            - class_ids [K] numeric class IDs
    """
    if target_class_ids is None:
        target_class_ids = {
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorcycle",
            6: "bus",
            8: "truck",
        }

    # Softmax over classes
    prob_scores = softmax(logits, axis=-1)

    cand_boxes_list: List[np.ndarray] = []
    cand_scores_list: List[np.ndarray] = []
    cand_labels_list: List[np.ndarray] = []

    # Filter by class and confidence BEFORE NMS
    for class_id, class_name in target_class_ids.items():
        if class_id >= prob_scores.shape[-1]:
            continue
        scores = prob_scores[:, class_id]
        thresh = class_thresholds.get(class_name, conf_threshold) if class_thresholds else conf_threshold
        mask = scores > thresh
        if not np.any(mask):
            continue

        selected_scores = scores[mask]
        selected_deltas = regressions[mask]
        selected_anchors = anchors[mask]

        decoded = decode_boxes(selected_deltas, selected_anchors)

        cand_boxes_list.append(decoded)
        cand_scores_list.append(selected_scores)
        cand_labels_list.append(np.full(len(selected_scores), class_id, dtype=np.int64))

    if not cand_boxes_list:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    all_boxes = np.concatenate(cand_boxes_list, axis=0)
    all_scores = np.concatenate(cand_scores_list, axis=0)
    all_labels = np.concatenate(cand_labels_list, axis=0)

    # Clip to image boundaries (0 to input_size)
    all_boxes[:, 0] = np.clip(all_boxes[:, 0], 0, input_size)
    all_boxes[:, 1] = np.clip(all_boxes[:, 1], 0, input_size)
    all_boxes[:, 2] = np.clip(all_boxes[:, 2], 0, input_size)
    all_boxes[:, 3] = np.clip(all_boxes[:, 3], 0, input_size)

    # Batched NMS: offset boxes per class so different classes never suppress each other initially
    offset_boxes = all_boxes + (all_labels[:, None].astype(np.float32) * (input_size * 4.0))
    keep = nms_vectorized(offset_boxes, all_scores, iou_threshold=nms_iou, max_detections=max_detections)

    kept_boxes = all_boxes[keep]
    kept_scores = all_scores[keep]
    kept_labels = all_labels[keep]

    # Normalize boxes to [0.0, 1.0]
    norm_boxes = kept_boxes / float(input_size)
    norm_boxes = np.clip(norm_boxes, 0.0, 1.0)

    # Apply cross-class vehicle deduplication if enabled
    if inter_class_vehicle_nms and len(norm_boxes) > 1:
        norm_boxes, kept_scores, kept_labels = deduplicate_cross_class_detections(
            boxes=norm_boxes,
            scores=kept_scores,
            labels=kept_labels,
            vehicle_ids=[2, 3, 4, 6, 8],
            iou_threshold=inter_class_iou,
        )

    return norm_boxes, kept_scores, kept_labels
