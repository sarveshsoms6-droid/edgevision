"""Unit tests for rolling median temporal count smoothing."""

import pytest
from edgevision.smoothing import apply_rolling_median_counts


def test_apply_rolling_median_counts_removes_spike():
    """Verify single-frame transient false positive spike is suppressed."""
    raw = [
        {"frame": 0, "car": 2, "person": 1},
        {"frame": 1, "car": 2, "person": 1},
        {"frame": 2, "car": 10, "person": 1},  # Spike
        {"frame": 3, "car": 2, "person": 1},
        {"frame": 4, "car": 2, "person": 1},
    ]

    smoothed = apply_rolling_median_counts(raw, window_size=5)
    assert len(smoothed) == 5
    assert smoothed[2]["car"] == 2
    assert smoothed[2]["person"] == 1


def test_apply_rolling_median_counts_removes_drop():
    """Verify single-frame transient false negative dropout is suppressed."""
    raw = [
        {"frame": 0, "car": 4},
        {"frame": 1, "car": 4},
        {"frame": 2, "car": 0},  # Transient drop
        {"frame": 3, "car": 4},
        {"frame": 4, "car": 4},
    ]

    smoothed = apply_rolling_median_counts(raw, window_size=5)
    assert smoothed[2]["car"] == 4


def test_apply_rolling_median_counts_preserves_step_transition():
    """Verify genuine sustained traffic changes are preserved."""
    raw = [
        {"frame": 0, "car": 1},
        {"frame": 1, "car": 1},
        {"frame": 2, "car": 1},
        {"frame": 3, "car": 5},
        {"frame": 4, "car": 5},
        {"frame": 5, "car": 5},
    ]

    smoothed = apply_rolling_median_counts(raw, window_size=3)
    assert smoothed[0]["car"] == 1
    assert smoothed[5]["car"] == 5


def test_apply_rolling_median_counts_edge_cases():
    """Verify empty and single-element edge cases."""
    assert apply_rolling_median_counts([]) == []
    assert apply_rolling_median_counts([{"frame": 0, "car": 3}]) == [{"frame": 0, "car": 3}]
