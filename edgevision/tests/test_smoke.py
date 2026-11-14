"""Smoke test suite verifying that all EdgeVision modules are importable
and the configuration loader correctly validates config.yaml.
"""

from __future__ import annotations

import importlib
import pytest
from pathlib import Path

MODULES = [
    "edgevision",
    "edgevision.config",
    "edgevision.video_io",
    "edgevision.preprocess",
    "edgevision.detector_pytorch",
    "edgevision.detector_et",
    "edgevision.detector_yolo",
    "edgevision.postprocess",
    "edgevision.counter",
    "edgevision.zones",
    "edgevision.smoothing",
    "edgevision.annotate",
    "edgevision.pipeline",
    "edgevision.benchmark",
    "edgevision.report",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str):
    """Verify that every module in the package imports without error."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_load_config_valid():
    """Verify that config.yaml loads and validates successfully."""
    from edgevision.config import load_config, EdgeVisionConfig

    # Find config.yaml
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = load_config(config_path)

    assert isinstance(cfg, EdgeVisionConfig)
    assert cfg.model.name in ("yolo11n", "ssdlite320_mobilenet_v3_large")
    assert cfg.model.input_size in (320, 640)
    assert cfg.model.runtime in (
        "yolov11",
        "yolov11n",
        "executorch-int8",
        "executorch-fp32",
        "pytorch-reference",
    )
    assert cfg.conf_threshold == 0.40
    assert cfg.nms_iou == 0.45
    assert cfg.max_detections == 100
    assert cfg.frame_stride == "auto"
    assert cfg.threads == 4
    assert cfg.smoothing.enabled is True
    assert cfg.smoothing.window == 5
    assert cfg.congestion.low_max == 5.0
    assert cfg.congestion.medium_max == 12.0
    assert cfg.privacy.blur_persons_in_samples is True
    assert cfg.upload.max_seconds == 60
    assert "mp4" in cfg.upload.allowed


def test_config_validation_failures():
    """Verify that invalid configuration values raise ValueError."""
    from edgevision.config import EdgeVisionConfig, ModelConfig, SmoothingConfig

    # Invalid runtime
    with pytest.raises(ValueError):
        cfg = EdgeVisionConfig(model=ModelConfig(runtime="invalid-runtime"))  # type: ignore
        cfg.validate()

    # Invalid confidence threshold (> 1.0)
    with pytest.raises(ValueError):
        cfg = EdgeVisionConfig(conf_threshold=1.5)
        cfg.validate()

    # Invalid smoothing window (even number)
    with pytest.raises(ValueError):
        cfg = EdgeVisionConfig(smoothing=SmoothingConfig(window=4))
        cfg.validate()
