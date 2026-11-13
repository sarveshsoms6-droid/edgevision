"""Unit tests for object and vehicle counting logic using synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest

from edgevision.counter import (
    FrameCounts,
    compute_vehicle_total,
    count_objects_in_frame,
)


def test_count_objects_in_frame():
    """Verify tallying of detected classes with synthetic class ID arrays."""
    id_map = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
    target_classes = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

    # 3 cars, 2 persons, 1 bus, 1 unknown (e.g. dog = 18)
    synthetic_ids = np.array([3, 1, 3, 6, 1, 3, 18], dtype=np.int64)

    counts = count_objects_in_frame(synthetic_ids, id_map, target_classes)

    assert counts["car"] == 3
    assert counts["person"] == 2
    assert counts["bus"] == 1
    assert counts["truck"] == 0
    assert counts["bicycle"] == 0
    assert counts["motorcycle"] == 0


def test_compute_vehicle_total():
    """Verify vehicle category summation across vehicle classes."""
    sample_counts = {
        "person": 5,
        "car": 4,
        "bus": 2,
        "motorcycle": 1,
        "bicycle": 2,
        "truck": 1,
    }
    # Vehicles = 4 + 2 + 1 + 2 + 1 = 10 (persons excluded)
    assert compute_vehicle_total(sample_counts) == 10


def test_frame_counts_to_csv_dict():
    """Verify serialization of FrameCounts to CSV dictionary."""
    fc = FrameCounts(
        frame_index=15,
        timestamp_seconds=0.5,
        counts_by_class={"car": 2, "person": 1, "truck": 1},
        total_vehicles=3,
        total_pedestrians=1,
    )
    d = fc.to_csv_dict()
    assert d["frame"] == 15
    assert d["timestamp_s"] == 0.5
    assert d["car"] == 2
    assert d["person"] == 1
    assert d["truck"] == 1
    assert d["bus"] == 0
    assert d["total_vehicles"] == 3
