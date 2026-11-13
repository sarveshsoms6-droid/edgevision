"""Unit tests for NumPy vectorised post-processing routines using synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest

from edgevision.postprocess import (
    decode_boxes,
    nms_vectorized,
    postprocess_detections,
    softmax,
)


def test_decode_boxes_synthetic():
    """Verify box decoding logic with synthetic anchor and delta values."""
    # Anchors: 2 boxes of size 20x20 centered at (20, 20) and (50, 50)
    anchors = np.array([
        [10.0, 10.0, 30.0, 30.0],
        [40.0, 40.0, 60.0, 60.0],
    ], dtype=np.float32)

    # Zero deltas -> decoded box must equal anchor box exactly
    zero_deltas = np.zeros((2, 4), dtype=np.float32)
    decoded = decode_boxes(zero_deltas, anchors)
    np.testing.assert_allclose(decoded, anchors, atol=1e-5)

    # Shift dx by 1.0 (with weight wx=10.0, dx_real = 0.1 * width = 2.0)
    shift_deltas = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    shifted = decode_boxes(shift_deltas, anchors)
    assert shifted[0, 0] == pytest.approx(12.0, abs=1e-4)
    assert shifted[0, 2] == pytest.approx(32.0, abs=1e-4)

    # Empty inputs
    empty_decoded = decode_boxes(np.empty((0, 4)), np.empty((0, 4)))
    assert empty_decoded.shape == (0, 4)


def test_nms_vectorized_synthetic():
    """Verify vectorised NMS suppression with synthetic overlapping and disjoint boxes."""
    # Two identical boxes, different scores
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [10.0, 10.0, 50.0, 50.0],
        [100.0, 100.0, 150.0, 150.0],  # Disjoint box
    ], dtype=np.float32)

    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms_vectorized(boxes, scores, iou_threshold=0.5, max_detections=10)

    # Should keep highest score of duplicate (idx 0) and disjoint box (idx 2)
    assert list(keep) == [0, 2]

    # Empty input
    assert len(nms_vectorized(np.empty((0, 4)), np.empty(0))) == 0


def test_softmax_numerical_stability():
    """Verify softmax numerical stability on extreme values."""
    large_logits = np.array([[1000.0, 1000.0, 1000.0]], dtype=np.float32)
    probs = softmax(large_logits)
    assert not np.isnan(probs).any()
    np.testing.assert_allclose(probs, [[1 / 3, 1 / 3, 1 / 3]], atol=1e-5)


def test_postprocess_detections_synthetic():
    """Verify full postprocessing pipeline with synthetic logits and anchors."""
    num_anchors = 10
    num_classes = 91

    # Anchors on a 320x320 grid
    anchors = np.zeros((num_anchors, 4), dtype=np.float32)
    for i in range(num_anchors):
        anchors[i] = [i * 20.0, i * 20.0, i * 20.0 + 30.0, i * 20.0 + 30.0]

    # Deltas zero
    regressions = np.zeros((num_anchors, 4), dtype=np.float32)

    # Logits: class 3 (car) high confidence for anchor 0
    # class 1 (person) high confidence for anchor 5
    logits = np.full((num_anchors, num_classes), -100.0, dtype=np.float32)
    logits[0, 3] = 10.0  # High car score
    logits[5, 1] = 10.0  # High person score

    target_map = {1: "person", 3: "car"}
    boxes, scores, labels = postprocess_detections(
        logits=logits,
        regressions=regressions,
        anchors=anchors,
        conf_threshold=0.50,
        nms_iou=0.45,
        max_detections=10,
        target_class_ids=target_map,
        input_size=320,
    )

    assert len(boxes) == 2
    assert set(labels) == {1, 3}
    # Normalized coordinates in [0, 1]
    assert (boxes >= 0.0).all() and (boxes <= 1.0).all()
    assert (scores > 0.5).all()


def test_deduplicate_cross_class_detections():
    """Verify that cross-class vehicle duplicates (e.g. car vs truck on same object) are suppressed."""
    from edgevision.postprocess import deduplicate_cross_class_detections

    # Two overlapping boxes with IoU ~ 0.95: one is car (class 3, score 0.8), one is truck (class 8, score 0.6)
    boxes = np.array([
        [0.2, 0.2, 0.5, 0.5],
        [0.21, 0.20, 0.51, 0.50],
        [0.7, 0.7, 0.9, 0.9],  # Separate car
    ], dtype=np.float32)

    scores = np.array([0.8, 0.6, 0.75], dtype=np.float32)
    labels = np.array([3, 8, 3], dtype=np.int64)

    clean_b, clean_s, clean_l = deduplicate_cross_class_detections(
        boxes=boxes,
        scores=scores,
        labels=labels,
        vehicle_ids=[2, 3, 4, 6, 8],
        iou_threshold=0.40,
    )

    # Should eliminate the lower-confidence truck duplicate (class 8)
    assert len(clean_b) == 2
    assert list(clean_l) == [3, 3]
    assert 8 not in clean_l


def test_class_specific_thresholds():
    """Verify that class-specific thresholds filter each class according to its configured cutoff."""
    num_anchors = 4
    num_classes = 91

    anchors = np.zeros((num_anchors, 4), dtype=np.float32)
    for i in range(num_anchors):
        anchors[i] = [i * 30.0, i * 30.0, i * 30.0 + 20.0, i * 30.0 + 20.0]

    regressions = np.zeros((num_anchors, 4), dtype=np.float32)

    # Logits calibrated:
    # Anchor 0: car (class 3) score ~ 0.15 (above car threshold 0.12, below default 0.25)
    # Anchor 1: person (class 1) score ~ 0.15 (below person threshold 0.22)
    # Anchor 2: person (class 1) score ~ 0.30 (above person threshold 0.22)
    # Anchor 3: car (class 3) score ~ 0.08 (below car threshold 0.12)
    logits = np.full((num_anchors, num_classes), -100.0, dtype=np.float32)
    # Logit calculation: if bg is 0, exp(0) = 1.0. For class score p = exp(L)/(exp(L) + exp(0)), L = ln(p/(1-p))
    logits[:, 0] = 0.0  # Background
    logits[0, 3] = np.log(0.15 / 0.85)
    logits[1, 1] = np.log(0.15 / 0.85)
    logits[2, 1] = np.log(0.30 / 0.70)
    logits[3, 3] = np.log(0.08 / 0.92)

    target_map = {1: "person", 3: "car"}
    class_thresholds = {"car": 0.12, "person": 0.22}

    boxes, scores, labels = postprocess_detections(
        logits=logits,
        regressions=regressions,
        anchors=anchors,
        conf_threshold=0.25,  # General fallback
        class_thresholds=class_thresholds,
        nms_iou=0.45,
        max_detections=10,
        target_class_ids=target_map,
        input_size=320,
    )

    # Anchor 0 (car ~0.15 >= 0.12): KEPT
    # Anchor 1 (person ~0.15 < 0.22): REJECTED
    # Anchor 2 (person ~0.30 >= 0.22): KEPT
    # Anchor 3 (car ~0.08 < 0.12): REJECTED
    assert len(boxes) == 2
    assert set(labels) == {1, 3}


