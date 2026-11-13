"""Visual annotation and privacy blurring module.

Routines for:
- Drawing bounding boxes with category labels and confidence percentages.
- Rendering on-screen count overlays (person, car, truck, etc.).
- Drawing ROI polygon boundaries if zone counting is enabled.
- Applying Gaussian blur to detected person bounding boxes for privacy compliance.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import cv2
import numpy as np

# Distinct BGR colours for target classes
CLASS_COLORS = {
    "person": (255, 105, 65),     # Royal blue-ish
    "car": (46, 204, 113),        # Emerald green
    "truck": (230, 126, 34),      # Orange
    "bus": (155, 89, 182),        # Amethyst purple
    "motorcycle": (52, 152, 219), # Sky blue
    "bicycle": (241, 196, 15),    # Sunflower yellow
}
DEFAULT_COLOR = (200, 200, 200)


def blur_person_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    person_class_id: int = 1,
) -> np.ndarray:
    """Apply Gaussian blur to regions corresponding to detected person boxes.

    Args:
        image: BGR image array (H, W, 3).
        boxes: Normalized bounding boxes [K, 4] [x1, y1, x2, y2].
        class_ids: Array of detected class IDs [K].
        person_class_id: COCO ID corresponding to person class.

    Returns:
        Image with blurred person regions.
    """
    out = image.copy()
    h, w = out.shape[:2]

    for box, cid in zip(boxes, class_ids):
        if int(cid) == person_class_id:
            x1 = max(0, int(box[0] * w))
            y1 = max(0, int(box[1] * h))
            x2 = min(w, int(box[2] * w))
            y2 = min(h, int(box[3] * h))

            if x2 > x1 and y2 > y1:
                roi = out[y1:y2, x1:x2]
                # Kernel size proportional to roi dimension (must be odd)
                kw = max(15, (roi.shape[1] // 2) * 2 + 1)
                kh = max(15, (roi.shape[0] // 2) * 2 + 1)
                blurred = cv2.GaussianBlur(roi, (kw, kh), sigmaX=30)
                out[y1:y2, x1:x2] = blurred

    return out


def annotate_frame(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_name_map: Dict[int, str],
    counts: Dict[str, int],
    zone_polygon: Optional[List[List[float]]] = None,
    blur_persons: bool = False,
) -> np.ndarray:
    """Render bounding boxes, confidence tags, count overlay, and optional privacy blur.

    Args:
        frame: BGR video frame array (H, W, 3).
        boxes: Bounding boxes [K, 4] in normalised coordinates [x1, y1, x2, y2].
        scores: Confidence scores [K].
        class_ids: Detected class IDs [K].
        class_name_map: Mapping of category ID to class label.
        counts: Dictionary of current frame counts.
        zone_polygon: Optional normalized polygon vertices for ROI display.
        blur_persons: Whether to apply privacy blurring to person detections.

    Returns:
        Annotated BGR image array.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # 1. Apply privacy blur first if enabled
    if blur_persons:
        out = blur_person_detections(out, boxes, class_ids, person_class_id=1)

    # 2. Draw ROI polygon if present
    if zone_polygon and len(zone_polygon) >= 3:
        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone_polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # 3. Draw bounding boxes and labels
    for box, score, cid in zip(boxes, scores, class_ids):
        cls_name = class_name_map.get(int(cid), f"id:{cid}")
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        x1 = max(0, int(box[0] * w))
        y1 = max(0, int(box[1] * h))
        x2 = min(w, int(box[2] * w))
        y2 = min(h, int(box[3] * h))

        # Box outline
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label badge
        label_text = f"{cls_name} {score * 100:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, scale, thick)
        cv2.rectangle(out, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label_text, (x1 + 2, y1 - baseline - 2), font, scale, (255, 255, 255), thick)

    # 4. Draw count overlay banner at top-left
    overlay_lines = []
    vehicle_sum = sum(counts.get(k, 0) for k in ("car", "truck", "bus", "motorcycle", "bicycle"))
    overlay_lines.append(f"Vehicles: {vehicle_sum} | Persons: {counts.get('person', 0)}")

    detail_str = " | ".join(f"{k}:{v}" for k, v in counts.items() if v > 0)
    if detail_str:
        overlay_lines.append(detail_str)

    # Render translucent banner background
    banner_h = 24 * len(overlay_lines) + 10
    banner_w = min(w - 20, 480)
    sub_img = out[10:10 + banner_h, 10:10 + banner_w]
    if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
        dark_rect = np.zeros_like(sub_img)
        cv2.addWeighted(sub_img, 0.4, dark_rect, 0.6, 0, sub_img)
        out[10:10 + banner_h, 10:10 + banner_w] = sub_img

    y_offset = 28
    for line in overlay_lines:
        cv2.putText(
            out,
            line,
            (18, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += 24

    return out
