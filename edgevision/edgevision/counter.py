"""Object and vehicle counting module.

Computes:
- Per-class counts for targeted categories (person, bicycle, car, motorcycle, bus, truck).
- Total vehicle count per frame and aggregated across the video.
- Structured count records for CSV and reporting, with optional zone metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}


@dataclass
class FrameCounts:
    """Detection counts for a single video frame."""

    frame_index: int
    timestamp_seconds: float
    counts_by_class: Dict[str, int] = field(default_factory=dict)
    total_vehicles: int = 0
    total_pedestrians: int = 0
    in_zone_vehicles: Optional[int] = None
    in_zone_pedestrians: Optional[int] = None

    def to_csv_dict(self) -> Dict[str, object]:
        """Format as flattened dictionary for CSV row output."""
        res: Dict[str, object] = {
            "frame": self.frame_index,
            "timestamp_s": round(self.timestamp_seconds, 3),
            "person": self.counts_by_class.get("person", 0),
            "bicycle": self.counts_by_class.get("bicycle", 0),
            "car": self.counts_by_class.get("car", 0),
            "motorcycle": self.counts_by_class.get("motorcycle", 0),
            "bus": self.counts_by_class.get("bus", 0),
            "truck": self.counts_by_class.get("truck", 0),
            "total_vehicles": self.total_vehicles,
        }
        if self.in_zone_vehicles is not None:
            res["in_zone_vehicles"] = self.in_zone_vehicles
        if self.in_zone_pedestrians is not None:
            res["in_zone_pedestrians"] = self.in_zone_pedestrians
        return res


def count_objects_in_frame(
    class_ids: np.ndarray,
    id_to_class_map: Dict[int, str],
    target_classes: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Tally counts for target classes from detected class IDs.

    Args:
        class_ids: Array of detected numeric class IDs [K].
        id_to_class_map: Mapping of numeric ID to class string.
        target_classes: List of target class names to include in counts dict.

    Returns:
        Dictionary mapping class name to integer count.
    """
    if target_classes is None:
        target_classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

    counts: Dict[str, int] = {cls_name: 0 for cls_name in target_classes}

    for cid in class_ids:
        cid_int = int(cid)
        cls_name = id_to_class_map.get(cid_int)
        if cls_name in counts:
            counts[cls_name] += 1

    return counts


def compute_vehicle_total(counts: Dict[str, int]) -> int:
    """Sum total vehicle count across all vehicular categories."""
    return sum(counts.get(cls_name, 0) for cls_name in VEHICLE_CLASSES)
