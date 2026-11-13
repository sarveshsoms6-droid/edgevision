"""Unit tests for report generation, congestion classification, and trend charts."""

import os
from pathlib import Path
import pytest

from edgevision.report import (
    determine_congestion_level,
    generate_planning_insight,
    generate_summary_markdown,
    plot_counts_over_time,
)


def test_determine_congestion_level():
    """Verify traffic congestion classification thresholds."""
    assert determine_congestion_level(2.5, low_max=5.0, medium_max=12.0) == "Low"
    assert determine_congestion_level(5.0, low_max=5.0, medium_max=12.0) == "Low"
    assert determine_congestion_level(8.0, low_max=5.0, medium_max=12.0) == "Medium"
    assert determine_congestion_level(12.0, low_max=5.0, medium_max=12.0) == "Medium"
    assert determine_congestion_level(15.0, low_max=5.0, medium_max=12.0) == "High"


def test_generate_planning_insight():
    """Verify rule-based insight generation logic."""
    insight_high = generate_planning_insight(14.0, 6.0, 2.33, "High")
    assert "crosswalks" in insight_high.lower() or "calming" in insight_high.lower()

    insight_low = generate_planning_insight(2.0, 1.0, 2.0, "Low")
    assert "capacity" in insight_low.lower()


def test_plot_counts_over_time_and_summary_markdown(tmp_path):
    """Verify PNG chart and summary markdown generation."""
    records = [
        {"frame": 0, "timestamp_s": 0.0, "total_vehicles": 3, "person": 1, "car": 3},
        {"frame": 1, "timestamp_s": 0.5, "total_vehicles": 4, "person": 2, "car": 4},
        {"frame": 2, "timestamp_s": 1.0, "total_vehicles": 2, "person": 1, "car": 2},
    ]

    chart_file = tmp_path / "counts_chart.png"
    plot_counts_over_time(records, chart_file)
    assert chart_file.is_file()
    assert chart_file.stat().st_size > 1000

    summary_file = tmp_path / "summary.md"
    payload = {
        "peak_counts": {"vehicles": 4, "person": 2, "car": 4},
        "average_counts": {"vehicles": 3.0, "person": 1.33, "car": 3.0},
        "congestion_level": "Low",
        "runtime_used": "executorch-int8",
        "total_frames": 3,
        "processed_frames": 3,
        "end_to_end_fps": 30.0,
        "processing_time_seconds": 0.1,
    }
    content = generate_summary_markdown(payload, summary_file)
    assert summary_file.is_file()
    assert "Low" in content
    assert "Total Vehicles" in content
