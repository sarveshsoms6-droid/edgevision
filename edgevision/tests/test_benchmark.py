"""Unit tests for benchmarking and telemetry utilities."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from edgevision.benchmark import (
    BenchmarkMetrics,
    calculate_latency_percentiles,
    get_peak_rss_mb,
    read_cpu_temperature,
)


def test_calculate_latency_percentiles():
    """Verify latency percentiles computation."""
    latencies = [0.010, 0.020, 0.030, 0.040, 0.050]  # 10ms to 50ms
    stats = calculate_latency_percentiles(latencies)

    assert stats["mean_ms"] == pytest.approx(30.0, abs=1e-3)
    assert stats["p50_ms"] == pytest.approx(30.0, abs=1e-3)
    assert stats["p95_ms"] == pytest.approx(48.0, abs=1.0)
    assert stats["p99_ms"] == pytest.approx(49.6, abs=1.0)

    # Empty list
    empty_stats = calculate_latency_percentiles([])
    assert empty_stats["mean_ms"] == 0.0


def test_read_cpu_temperature_graceful():
    """Verify read_cpu_temperature returns float or None without throwing."""
    temp = read_cpu_temperature()
    assert temp is None or isinstance(temp, float)


def test_get_peak_rss_mb():
    """Verify peak RSS returns non-zero float."""
    rss = get_peak_rss_mb()
    assert isinstance(rss, float)
    assert rss > 0.0


def test_benchmark_metrics_json_export(tmp_path):
    """Verify BenchmarkMetrics serialization to JSON file."""
    metrics = BenchmarkMetrics(
        runtime_used="executorch-fp32",
        num_processed_frames=60,
        num_skipped_frames=60,
        total_time_seconds=2.0,
        end_to_end_fps=60.0,
        effective_fps=30.0,
        latency_mean_ms=33.3,
        latency_p50_ms=32.0,
        latency_p95_ms=38.0,
        latency_p99_ms=40.0,
        peak_rss_mb=120.5,
        cpu_temp_start_c=None,
        cpu_temp_end_c=None,
        model_size_mb=13.15,
    )

    out_file = tmp_path / "benchmark.json"
    metrics.save_json(out_file)
    assert out_file.is_file()

    with open(out_file, "r") as f:
        data = json.load(f)

    assert data["runtime_used"] == "executorch-fp32"
    assert data["num_processed_frames"] == 60
    assert data["latency_p50_ms"] == 32.0
    assert data["peak_rss_mb"] == 120.5
