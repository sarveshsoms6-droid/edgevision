"""Performance benchmarking and system telemetry module.

Measures:
- Latency statistics (p50, p95, p99, mean) per processed frame.
- Throughput (effective FPS including skipped frames vs end-to-end FPS).
- Peak resident set size (RSS) memory using psutil.
- CPU temperature via Raspberry Pi `vcgencmd measure_temp` (returns None gracefully on non-Pi platforms).
- Runtime provenance metadata recording.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Dict, List, Optional
import numpy as np
import psutil


@dataclass
class BenchmarkMetrics:
    """Benchmark metrics container."""

    runtime_used: str
    num_processed_frames: int
    num_skipped_frames: int
    total_time_seconds: float
    end_to_end_fps: float
    effective_fps: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    peak_rss_mb: float
    cpu_temp_start_c: Optional[float]
    cpu_temp_end_c: Optional[float]
    model_size_mb: Optional[float] = None
    device_platform: str = platform.platform()
    cpu_architecture: str = platform.machine()
    cpu_model: str = platform.processor() or "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for benchmark.json."""
        return {
            "runtime_used": self.runtime_used,
            "device_platform": self.device_platform,
            "cpu_architecture": self.cpu_architecture,
            "cpu_model": self.cpu_model,
            "num_processed_frames": self.num_processed_frames,
            "num_skipped_frames": self.num_skipped_frames,
            "total_time_seconds": round(self.total_time_seconds, 3),
            "end_to_end_fps": round(self.end_to_end_fps, 2),
            "effective_fps": round(self.effective_fps, 2),
            "latency_mean_ms": round(self.latency_mean_ms, 2),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "latency_p99_ms": round(self.latency_p99_ms, 2),
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "cpu_temp_start_c": round(self.cpu_temp_start_c, 1) if self.cpu_temp_start_c is not None else None,
            "cpu_temp_end_c": round(self.cpu_temp_end_c, 1) if self.cpu_temp_end_c is not None else None,
            "model_size_mb": round(self.model_size_mb, 2) if self.model_size_mb is not None else None,
        }

    def save_json(self, output_path: str | Path) -> None:
        """Write metrics to benchmark.json."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def read_cpu_temperature() -> Optional[float]:
    """Read CPU core temperature via vcgencmd or sysfs (graceful fallback on non-Pi).

    Returns:
        Float temperature in Celsius, or None if unavailable.
    """
    # 1. Try vcgencmd on Raspberry Pi OS
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL, timeout=0.5).decode("utf-8")
        match = re.search(r"temp=([\d\.]+)", out)
        if match:
            return float(match.group(1))
    except Exception:
        pass

    # 2. Try Linux sysfs thermal zone
    sysfs_temp = Path("/sys/class/thermal/thermal_zone0/temp")
    if sysfs_temp.is_file():
        try:
            val = sysfs_temp.read_text().strip()
            return float(val) / 1000.0
        except Exception:
            pass

    return None


def get_peak_rss_mb() -> float:
    """Read current process Resident Set Size (RSS) in megabytes."""
    try:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def calculate_latency_percentiles(latencies_seconds: List[float]) -> Dict[str, float]:
    """Calculate mean, p50, p95, and p99 latency in milliseconds.

    Args:
        latencies_seconds: List of recorded inference durations in seconds.

    Returns:
        Dictionary of latency statistics in milliseconds.
    """
    if not latencies_seconds:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }

    ms_array = np.array(latencies_seconds, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(np.mean(ms_array)),
        "p50_ms": float(np.percentile(ms_array, 50)),
        "p95_ms": float(np.percentile(ms_array, 95)),
        "p99_ms": float(np.percentile(ms_array, 99)),
    }


from edgevision.stream_utils import safe_print


def print_benchmark_table(metrics: BenchmarkMetrics) -> None:
    """Print formatted ASCII benchmark results table."""
    d = metrics.to_dict()
    temp_str = (
        f"{d['cpu_temp_start_c']} 'C -> {d['cpu_temp_end_c']} 'C"
        if d['cpu_temp_start_c'] is not None
        else "NOT AVAILABLE (Non-Pi Host)"
    )

    safe_print("\n" + "=" * 70)
    safe_print("EDGEVISION PERFORMANCE BENCHMARK RESULTS")
    safe_print("=" * 70)
    safe_print(f"Runtime Used         : {d['runtime_used']}")
    safe_print(f"Platform / Host      : {d['device_platform']} ({d['cpu_architecture']})")
    safe_print(f"CPU Model            : {d['cpu_model']}")
    safe_print(f"Model File Size      : {d['model_size_mb']} MB" if d['model_size_mb'] else "Model File Size      : N/A")
    safe_print(f"Total Frames Processed: {d['num_processed_frames']} (Skipped: {d['num_skipped_frames']})")
    safe_print(f"Total Time           : {d['total_time_seconds']} s")
    safe_print(f"End-to-End FPS       : {d['end_to_end_fps']} FPS")
    safe_print(f"Effective Ingest FPS : {d['effective_fps']} FPS")
    safe_print("-" * 70)
    safe_print("INFERENCE LATENCY BREAKDOWN (per processed frame):")
    safe_print(f"  Mean Latency       : {d['latency_mean_ms']:.2f} ms")
    safe_print(f"  p50 (Median)       : {d['latency_p50_ms']:.2f} ms")
    safe_print(f"  p95                : {d['latency_p95_ms']:.2f} ms")
    safe_print(f"  p99                : {d['latency_p99_ms']:.2f} ms")
    safe_print("-" * 70)
    safe_print("SYSTEM TELEMETRY:")
    safe_print(f"  Peak RSS Memory    : {d['peak_rss_mb']:.1f} MB")
    safe_print(f"  CPU Temperature    : {temp_str}")
    safe_print("=" * 70 + "\n")
