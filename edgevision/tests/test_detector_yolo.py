"""Unit tests for the YOLOv11 detector integration."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from edgevision.detector_yolo import YOLOv11Detector


@pytest.fixture
def yolo_weights_path() -> Path:
    """Resolve path to local yolo11n.pt model file."""
    candidates = [
        Path("models/yolo11n.pt"),
        Path("edgevision/models/yolo11n.pt"),
        Path(__file__).resolve().parent.parent / "models" / "yolo11n.pt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    pytest.skip("yolo11n.pt not found in local models directory")


def test_yolo_detector_init(yolo_weights_path: Path):
    """Verify YOLOv11 detector initializes properly with class mappings."""
    detector = YOLOv11Detector(model_path=yolo_weights_path, conf_threshold=0.30)
    assert detector.runtime_name == "yolov11"
    assert isinstance(detector.id_to_class_map, dict)
    assert len(detector.id_to_class_map) > 0
    # COCO classes check
    assert detector.id_to_class_map.get(0) == "person"
    assert detector.id_to_class_map.get(2) == "car"


def test_yolo_detector_warmup(yolo_weights_path: Path):
    """Verify warm-up inference completes without error."""
    detector = YOLOv11Detector(model_path=yolo_weights_path)
    detector.warmup(num_iterations=2, input_size=320)


def test_yolo_detector_predict_blank_image(yolo_weights_path: Path):
    """Verify inference on blank image returns empty or valid coordinate structures."""
    detector = YOLOv11Detector(model_path=yolo_weights_path, conf_threshold=0.5)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes, scores, labels = detector.predict_detections(blank)

    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert boxes.ndim == 2
    assert boxes.shape[1] == 4 or boxes.shape[0] == 0
    if len(boxes) > 0:
        assert np.all(boxes >= 0.0) and np.all(boxes <= 1.0)
        assert len(scores) == len(boxes)
        assert len(labels) == len(boxes)


def test_yolo_detector_predict_target_filter(yolo_weights_path: Path):
    """Verify target_classes filter retains only specified classes."""
    detector = YOLOv11Detector(model_path=yolo_weights_path, conf_threshold=0.1)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes, scores, labels = detector.predict_detections(
        blank,
        target_classes=["car", "bus", "truck"],
    )
    for cid in labels:
        assert detector.id_to_class_map.get(int(cid)) in {"car", "bus", "truck"}


def test_yolo_detector_missing_model():
    """Verify FileNotFoundError is raised for non-existent model files."""
    with pytest.raises(FileNotFoundError):
        _ = YOLOv11Detector(model_path="non_existent_model_weights.pt")


def test_pipeline_e2e_with_yolov11(tmp_path: Path):
    """Verify EdgeVisionPipeline runs end-to-end with YOLOv11 runtime."""
    from edgevision.config import load_config
    from edgevision.pipeline import EdgeVisionPipeline

    sample_video = Path(__file__).resolve().parent.parent / "sample_data" / "traffic_sample.mp4"
    if not sample_video.is_file():
        pytest.skip("traffic_sample.mp4 not found")

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    cfg = load_config(config_path)
    cfg.model.runtime = "yolov11"
    cfg.frame_stride = 2
    cfg.output.save_video = True
    cfg.output.save_samples = 2

    pipeline = EdgeVisionPipeline(cfg)
    res = pipeline.process_video(
        video_path=sample_video,
        output_dir=tmp_path / "yolo_out",
        runtime_override="yolov11",
    )

    assert res.runtime_used == "yolov11"
    assert res.total_frames > 0
    assert res.processed_frames > 0
    assert Path(res.csv_path).is_file()
    assert Path(res.benchmark_json_path).is_file()
    assert Path(res.summary_md_path).is_file()
    assert Path(res.chart_path).is_file()
    assert Path(res.video_output_path).is_file()

