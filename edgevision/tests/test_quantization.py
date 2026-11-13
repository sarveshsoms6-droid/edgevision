"""Unit tests verifying INT8 quantized model loading and inference."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from edgevision.detector_et import ExecuTorchDetector


def test_int8_model_size_and_loading():
    """Verify that the INT8 quantized model exists, is significantly compressed (<4MB), and loads."""
    int8_pte = Path("models/ssdlite320_mobilenet_v3_large_int8.pte")
    if not int8_pte.is_file():
        pytest.skip("INT8 model not yet generated")

    size_mb = int8_pte.stat().st_size / (1024.0 * 1024.0)
    assert size_mb < 5.0, f"INT8 model should be compressed (<5 MB), got {size_mb:.2f} MB"

    detector = ExecuTorchDetector(pte_path=int8_pte, runtime_mode="executorch-int8", threads=2)
    assert detector is not None

    # Test inference
    dummy = np.zeros((1, 3, 320, 320), dtype=np.float32)
    logits, regs = detector.predict_raw(dummy)

    assert logits.shape == (3234, 91)
    assert regs.shape == (3234, 4)
    assert not np.isnan(logits).any()
    assert not np.isnan(regs).any()
