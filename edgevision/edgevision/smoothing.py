"""Temporal count smoothing module.

Implements:
- Rolling median filter applied across consecutive frame counts to suppress jitter
  and transient false positives/negatives.
- Border/edge handling for initial and final frames in the video sequence.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def apply_rolling_median_counts(
    counts_sequence: List[Dict[str, Any]],
    window_size: int = 5,
) -> List[Dict[str, Any]]:
    """Apply rolling median filter to a sequence of per-frame class count dictionaries.

    Args:
        counts_sequence: Chronological list of {class_name: count} dictionaries.
        window_size: Odd integer specifying the filter window width.

    Returns:
        New sequence of count dictionaries smoothed with the rolling median filter.
    """
    if not counts_sequence:
        return []

    if window_size <= 1 or len(counts_sequence) == 1:
        return [dict(d) for d in counts_sequence]

    half_w = max(1, window_size // 2)
    n = len(counts_sequence)

    # Identify numeric keys to smooth
    numeric_keys = [
        k for k, v in counts_sequence[0].items()
        if isinstance(v, (int, float, np.integer, np.floating)) and k not in ("frame", "timestamp_s")
    ]

    # Pre-extract arrays for fast NumPy slicing
    key_arrays: Dict[str, np.ndarray] = {
        k: np.array([row.get(k, 0) for row in counts_sequence], dtype=np.float32)
        for k in numeric_keys
    }

    smoothed_sequence: List[Dict[str, Any]] = []

    for i in range(n):
        start_idx = max(0, i - half_w)
        end_idx = min(n, i + half_w + 1)

        row_copy = dict(counts_sequence[i])

        for k in numeric_keys:
            slice_vals = key_arrays[k][start_idx:end_idx]
            med_val = float(np.median(slice_vals))
            row_copy[k] = int(round(med_val))

        smoothed_sequence.append(row_copy)

    return smoothed_sequence
