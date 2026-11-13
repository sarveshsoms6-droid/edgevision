"""Polygon Region of Interest (ROI) counting module.

Implements:
- Point-in-polygon algorithm using Ray Casting and box centers.
- Filtering detections within configured ROI polygons.
- In-zone vehicle and pedestrian count computation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}


def is_point_in_polygon(
    point: Tuple[float, float] | Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> bool:
    """Check if a 2D point [x, y] is inside a polygon using ray casting.

    Args:
        point: Coordinates (x, y).
        polygon: List of polygon vertices [(x1, y1), (x2, y2), ...].

    Returns:
        True if point is strictly inside or on the boundary of the polygon.
    """
    n = len(polygon)
    if n < 3:
        return False

    x, y = float(point[0]), float(point[1])
    inside = False

    p1x, p1y = float(polygon[0][0]), float(polygon[0][1])
    for i in range(1, n + 1):
        p2 = polygon[i % n]
        p2x, p2y = float(p2[0]), float(p2[1])

        # Check ray crossing
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
                elif p1x == p2x or x <= p1x:
                    inside = not inside

        p1x, p1y = p2x, p2y

    return inside


def filter_boxes_in_zone(
    boxes: np.ndarray,
    polygon: Sequence[Sequence[float]],
) -> np.ndarray:
    """Filter bounding boxes whose center points fall inside the ROI polygon.

    Args:
        boxes: Bounding boxes [K, 4] in normalized coordinates [x1, y1, x2, y2].
        polygon: List of vertices [[x, y], ...] in normalized coordinates.

    Returns:
        Boolean mask array of shape [K] indicating which boxes fall inside the zone.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)

    if len(polygon) < 3:
        return np.zeros(len(boxes), dtype=bool)

    # Box centers in normalized coordinates
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

    mask = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        mask[i] = is_point_in_polygon((cx[i], cy[i]), polygon)

    return mask


def count_in_zone(
    boxes: np.ndarray,
    labels: np.ndarray,
    polygon: Sequence[Sequence[float]],
    id_to_class_map: Dict[int, str],
) -> Dict[str, int]:
    """Calculate target class counts specifically for objects within the ROI zone.

    Args:
        boxes: Normalized bounding boxes [K, 4].
        labels: Class IDs [K].
        polygon: Polygon vertices [[x, y], ...].
        id_to_class_map: Mapping of numeric class ID to class label.

    Returns:
        Dictionary with in-zone counts:
        - per target class counts
        - in_zone_vehicles: total vehicles in zone
        - in_zone_pedestrians: total persons in zone
    """
    res: Dict[str, int] = {
        "person": 0,
        "bicycle": 0,
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
        "in_zone_vehicles": 0,
        "in_zone_pedestrians": 0,
    }

    if len(boxes) == 0 or len(polygon) < 3:
        return res

    mask = filter_boxes_in_zone(boxes, polygon)
    zone_labels = labels[mask]

    for cid in zone_labels:
        cls_name = id_to_class_map.get(int(cid))
        if cls_name in res:
            res[cls_name] += 1
            if cls_name == "person":
                res["in_zone_pedestrians"] += 1
            elif cls_name in VEHICLE_CLASSES:
                res["in_zone_vehicles"] += 1

    return res
