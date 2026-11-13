"""Unit tests for polygon Region of Interest (ROI) algorithms."""

import pytest
import numpy as np

from edgevision.zones import count_in_zone, filter_boxes_in_zone, is_point_in_polygon


def test_is_point_in_polygon_inside_and_outside():
    """Verify ray casting logic on convex and non-trivial polygons."""
    poly = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]

    assert is_point_in_polygon((0.5, 0.5), poly) is True
    assert is_point_in_polygon((0.25, 0.25), poly) is True
    assert is_point_in_polygon((0.1, 0.1), poly) is False
    assert is_point_in_polygon((0.9, 0.5), poly) is False
    assert is_point_in_polygon((0.5, 0.9), poly) is False


def test_filter_boxes_in_zone():
    """Verify box filtering using box center coordinates."""
    poly = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]

    # Box 0: Center is (0.5, 0.5) -> Inside
    # Box 1: Center is (0.1, 0.1) -> Outside
    # Box 2: Center is (0.7, 0.7) -> Inside
    boxes = np.array([
        [0.4, 0.4, 0.6, 0.6],
        [0.05, 0.05, 0.15, 0.15],
        [0.65, 0.65, 0.75, 0.75],
    ], dtype=np.float32)

    mask = filter_boxes_in_zone(boxes, poly)
    assert mask.shape == (3,)
    assert mask.tolist() == [True, False, True]


def test_filter_boxes_in_zone_empty():
    """Verify behavior on empty box input."""
    poly = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    mask = filter_boxes_in_zone(empty_boxes, poly)
    assert len(mask) == 0


def test_count_in_zone():
    """Verify in-zone object and vehicle aggregation."""
    poly = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    coco_map = {1: "person", 3: "car", 8: "truck"}

    boxes = np.array([
        [0.4, 0.4, 0.6, 0.6],     # Center (0.5, 0.5) inside -> person
        [0.05, 0.05, 0.15, 0.15], # Center (0.1, 0.1) outside -> car
        [0.6, 0.6, 0.7, 0.7],     # Center (0.65, 0.65) inside -> truck
    ], dtype=np.float32)
    labels = np.array([1, 3, 8], dtype=np.int64)

    res = count_in_zone(boxes, labels, poly, coco_map)
    assert res["person"] == 1
    assert res["truck"] == 1
    assert res["car"] == 0
    assert res["in_zone_vehicles"] == 1  # truck is vehicle
    assert res["in_zone_pedestrians"] == 1  # person
