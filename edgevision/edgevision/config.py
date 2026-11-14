"""Configuration loader and schema validator for EdgeVision.

Loads config.yaml and validates all parameters, types, value bounds,
and structural constraints required for offline video processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Union
import yaml

VALID_RUNTIMES = (
    "yolov11",
    "yolov11n",
    "executorch-int8",
    "executorch-fp32",
    "pytorch-reference",
)


@dataclass
class ModelConfig:
    """Model architecture, weights, and runtime settings."""

    name: str = "yolo11n"
    pte_path: str = "models/ssdlite320_mobilenet_v3_large.pte"
    yolo_path: str = "models/yolo11n.pt"
    anchors_path: str = "models/anchors.npy"
    input_size: int = 640
    runtime: Literal[
        "yolov11",
        "yolov11n",
        "executorch-int8",
        "executorch-fp32",
        "pytorch-reference",
    ] = "yolov11"

    def validate(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"Invalid model name: {self.name!r}")
        if not isinstance(self.pte_path, str):
            raise ValueError(f"Invalid model pte_path: {self.pte_path!r}")
        if not isinstance(self.anchors_path, str):
            raise ValueError(f"Invalid model anchors_path: {self.anchors_path!r}")
        if hasattr(self, "yolo_path") and not isinstance(self.yolo_path, str):
            raise ValueError(f"Invalid model yolo_path: {self.yolo_path!r}")
        if not isinstance(self.input_size, int) or self.input_size <= 0:
            raise ValueError(
                f"model.input_size must be a positive integer, got {self.input_size!r}"
            )
        if self.runtime not in VALID_RUNTIMES:
            raise ValueError(
                f"Invalid model.runtime: {self.runtime!r}. Expected one of {VALID_RUNTIMES}"
            )


@dataclass
class ClassesConfig:
    """Target classes and COCO category mappings."""

    target_names: List[str] = field(
        default_factory=lambda: [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "truck",
            "ambulance",
        ]
    )
    coco_id_map: Dict[int, str] = field(
        default_factory=lambda: {
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorcycle",
            6: "bus",
            8: "truck",
        }
    )

    def validate(self) -> None:
        if not isinstance(self.target_names, list) or not self.target_names:
            raise ValueError("classes.target_names must be a non-empty list of strings")
        for name in self.target_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Invalid class name in target_names: {name!r}")
        if not isinstance(self.coco_id_map, dict) or not self.coco_id_map:
            raise ValueError("classes.coco_id_map must be a non-empty mapping of int->str")
        for k, v in self.coco_id_map.items():
            if not isinstance(k, int):
                raise ValueError(f"coco_id_map key must be an integer COCO ID, got {k!r}")
            if not isinstance(v, str):
                raise ValueError(f"coco_id_map value must be a string class name, got {v!r}")


@dataclass
class SmoothingConfig:
    """Temporal rolling median filter settings."""

    enabled: bool = True
    window: int = 5

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"smoothing.enabled must be a boolean, got {self.enabled!r}")
        if not isinstance(self.window, int) or self.window < 1:
            raise ValueError(
                f"smoothing.window must be a positive integer, got {self.window!r}"
            )
        if self.window % 2 == 0:
            raise ValueError(
                f"smoothing.window must be an odd integer, got {self.window}"
            )


@dataclass
class ZoneConfig:
    """Region of Interest (ROI) polygon configuration."""

    enabled: bool = False
    polygon: List[List[float]] = field(
        default_factory=lambda: [
            [0.10, 0.20],
            [0.90, 0.20],
            [0.90, 0.80],
            [0.10, 0.80],
        ]
    )

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"zone.enabled must be a boolean, got {self.enabled!r}")
        if not isinstance(self.polygon, list):
            raise ValueError(f"zone.polygon must be a list of vertices, got {type(self.polygon)}")
        if self.enabled:
            if len(self.polygon) < 3:
                raise ValueError(
                    f"zone.polygon must contain at least 3 vertices when enabled, got {len(self.polygon)}"
                )
            for pt in self.polygon:
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    raise ValueError(f"Each zone.polygon vertex must be [x, y], got {pt!r}")
                x, y = pt
                if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
                    raise ValueError(
                        f"Zone polygon coordinates must be normalised in [0.0, 1.0], got [{x}, {y}]"
                    )


@dataclass
class CongestionConfig:
    """Congestion assessment vehicle count thresholds."""

    low_max: float = 5.0
    medium_max: float = 12.0

    def validate(self) -> None:
        if not isinstance(self.low_max, (int, float)) or self.low_max < 0:
            raise ValueError(
                f"congestion.low_max must be a non-negative number, got {self.low_max!r}"
            )
        if not isinstance(self.medium_max, (int, float)) or self.medium_max < self.low_max:
            raise ValueError(
                f"congestion.medium_max must be >= low_max ({self.low_max}), got {self.medium_max!r}"
            )


@dataclass
class PrivacyConfig:
    """Privacy preservation options."""

    blur_persons_in_samples: bool = True

    def validate(self) -> None:
        if not isinstance(self.blur_persons_in_samples, bool):
            raise ValueError(
                f"privacy.blur_persons_in_samples must be a boolean, got {self.blur_persons_in_samples!r}"
            )


@dataclass
class OutputConfig:
    """Output artifacts and reporting configurations."""

    save_samples: int = 6
    save_video: bool = False
    csv: bool = True

    def validate(self) -> None:
        if not isinstance(self.save_samples, int) or self.save_samples < 0:
            raise ValueError(
                f"output.save_samples must be a non-negative integer, got {self.save_samples!r}"
            )
        if not isinstance(self.save_video, bool):
            raise ValueError(f"output.save_video must be a boolean, got {self.save_video!r}")
        if not isinstance(self.csv, bool):
            raise ValueError(f"output.csv must be a boolean, got {self.csv!r}")


@dataclass
class UploadConfig:
    """Video upload constraints for web interface."""

    max_seconds: int = 60
    max_mb: int = 100
    allowed: List[str] = field(
        default_factory=lambda: ["mp4", "avi", "mov", "mkv"]
    )

    def validate(self) -> None:
        if not isinstance(self.max_seconds, int) or self.max_seconds <= 0:
            raise ValueError(
                f"upload.max_seconds must be a positive integer, got {self.max_seconds!r}"
            )
        if not isinstance(self.max_mb, int) or self.max_mb <= 0:
            raise ValueError(
                f"upload.max_mb must be a positive integer, got {self.max_mb!r}"
            )
        if not isinstance(self.allowed, list) or not self.allowed:
            raise ValueError("upload.allowed must be a non-empty list of string extensions")


@dataclass
class EdgeVisionConfig:
    """Root configuration object for EdgeVision."""

    model: ModelConfig = field(default_factory=ModelConfig)
    classes: ClassesConfig = field(default_factory=ClassesConfig)
    conf_threshold: float = 0.20
    class_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "person": 0.22,
            "bicycle": 0.15,
            "car": 0.12,
            "motorcycle": 0.15,
            "bus": 0.14,
            "truck": 0.14,
        }
    )
    nms_iou: float = 0.35
    inter_class_vehicle_nms: bool = True
    inter_class_iou: float = 0.40
    max_detections: int = 100
    frame_stride: Union[int, Literal["auto"]] = "auto"
    threads: int = 4
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    zone: ZoneConfig = field(default_factory=ZoneConfig)
    congestion: CongestionConfig = field(default_factory=CongestionConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)

    def validate(self) -> None:
        """Validate all configuration sections and parameters."""
        self.model.validate()
        self.classes.validate()

        if not isinstance(self.conf_threshold, (int, float)) or not (
            0.0 <= float(self.conf_threshold) <= 1.0
        ):
            raise ValueError(
                f"conf_threshold must be a float in [0.0, 1.0], got {self.conf_threshold!r}"
            )

        if not isinstance(self.class_thresholds, dict):
            raise ValueError(
                f"class_thresholds must be a dictionary, got {type(self.class_thresholds)}"
            )

        if not isinstance(self.nms_iou, (int, float)) or not (
            0.0 <= float(self.nms_iou) <= 1.0
        ):
            raise ValueError(
                f"nms_iou must be a float in [0.0, 1.0], got {self.nms_iou!r}"
            )

        if not isinstance(self.inter_class_vehicle_nms, bool):
            raise ValueError(
                f"inter_class_vehicle_nms must be a boolean, got {self.inter_class_vehicle_nms!r}"
            )

        if not isinstance(self.inter_class_iou, (int, float)) or not (
            0.0 <= float(self.inter_class_iou) <= 1.0
        ):
            raise ValueError(
                f"inter_class_iou must be a float in [0.0, 1.0], got {self.inter_class_iou!r}"
            )

        if not isinstance(self.max_detections, int) or self.max_detections <= 0:
            raise ValueError(
                f"max_detections must be a positive integer, got {self.max_detections!r}"
            )

        if self.frame_stride != "auto":
            if not isinstance(self.frame_stride, int) or self.frame_stride < 1:
                raise ValueError(
                    f"frame_stride must be 'auto' or an integer >= 1, got {self.frame_stride!r}"
                )

        if not isinstance(self.threads, int) or not (1 <= self.threads <= 64):
            raise ValueError(
                f"threads must be an integer between 1 and 64, got {self.threads!r}"
            )

        self.smoothing.validate()
        self.zone.validate()
        self.congestion.validate()
        self.privacy.validate()
        self.output.validate()
        self.upload.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to nested dictionary."""
        return {
            "model": {
                "name": self.model.name,
                "pte_path": self.model.pte_path,
                "anchors_path": self.model.anchors_path,
                "input_size": self.model.input_size,
                "runtime": self.model.runtime,
            },
            "classes": {
                "target_names": list(self.classes.target_names),
                "coco_id_map": dict(self.classes.coco_id_map),
            },
            "conf_threshold": float(self.conf_threshold),
            "nms_iou": float(self.nms_iou),
            "max_detections": int(self.max_detections),
            "frame_stride": self.frame_stride,
            "threads": int(self.threads),
            "smoothing": {
                "enabled": self.smoothing.enabled,
                "window": self.smoothing.window,
            },
            "zone": {
                "enabled": self.zone.enabled,
                "polygon": [list(pt) for pt in self.zone.polygon],
            },
            "congestion": {
                "low_max": float(self.congestion.low_max),
                "medium_max": float(self.congestion.medium_max),
            },
            "privacy": {
                "blur_persons_in_samples": self.privacy.blur_persons_in_samples,
            },
            "output": {
                "save_samples": self.output.save_samples,
                "save_video": self.output.save_video,
                "csv": self.output.csv,
            },
            "upload": {
                "max_seconds": self.upload.max_seconds,
                "max_mb": self.upload.max_mb,
                "allowed": list(self.upload.allowed),
            },
        }


def load_config(config_path: Union[str, Path, None] = None) -> EdgeVisionConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, looks in current directory
                     or parent directory.

    Returns:
        Validated EdgeVisionConfig instance.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        ValueError: If configuration values fail schema validation.
    """
    if config_path is None:
        candidates = [
            Path("config.yaml"),
            Path("edgevision/config.yaml"),
            Path(__file__).resolve().parent.parent / "config.yaml",
        ]
        found: Path | None = None
        for cand in candidates:
            if cand.is_file():
                found = cand
                break
        if found is None:
            raise FileNotFoundError(
                "config.yaml not found in candidate paths: "
                f"{[str(p) for p in candidates]}"
            )
        target_path = found
    else:
        target_path = Path(config_path)
        if not target_path.is_file():
            raise FileNotFoundError(f"Config file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model_data = data.get("model", {})
    model_cfg = ModelConfig(
        name=model_data.get("name", "ssdlite320_mobilenet_v3_large"),
        pte_path=model_data.get("pte_path", "models/ssdlite320_mobilenet_v3_large.pte"),
        anchors_path=model_data.get("anchors_path", "models/anchors.npy"),
        input_size=int(model_data.get("input_size", 320)),
        runtime=model_data.get("runtime", "executorch-int8"),
    )

    classes_data = data.get("classes", {})
    # Handle both target_names and raw coco_id_map (ensuring integer keys)
    raw_coco_map = classes_data.get("coco_id_map", {})
    coco_id_map = {int(k): str(v) for k, v in raw_coco_map.items()} if raw_coco_map else {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        6: "bus",
        8: "truck",
    }
    classes_cfg = ClassesConfig(
        target_names=classes_data.get(
            "target_names",
            ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
        ),
        coco_id_map=coco_id_map,
    )

    smoothing_data = data.get("smoothing", {})
    smoothing_cfg = SmoothingConfig(
        enabled=bool(smoothing_data.get("enabled", True)),
        window=int(smoothing_data.get("window", 5)),
    )

    zone_data = data.get("zone", {})
    zone_cfg = ZoneConfig(
        enabled=bool(zone_data.get("enabled", False)),
        polygon=[[float(coord) for coord in pt] for pt in zone_data.get("polygon", [])]
        if "polygon" in zone_data
        else [
            [0.10, 0.20],
            [0.90, 0.20],
            [0.90, 0.80],
            [0.10, 0.80],
        ],
    )

    congestion_data = data.get("congestion", {})
    congestion_cfg = CongestionConfig(
        low_max=float(congestion_data.get("low_max", 5.0)),
        medium_max=float(congestion_data.get("medium_max", 12.0)),
    )

    privacy_data = data.get("privacy", {})
    privacy_cfg = PrivacyConfig(
        blur_persons_in_samples=bool(privacy_data.get("blur_persons_in_samples", True)),
    )

    output_data = data.get("output", {})
    output_cfg = OutputConfig(
        save_samples=int(output_data.get("save_samples", 6)),
        save_video=bool(output_data.get("save_video", False)),
        csv=bool(output_data.get("csv", True)),
    )

    upload_data = data.get("upload", {})
    upload_cfg = UploadConfig(
        max_seconds=int(upload_data.get("max_seconds", 60)),
        max_mb=int(upload_data.get("max_mb", 100)),
        allowed=[str(ext).lower().lstrip(".") for ext in upload_data.get("allowed", ["mp4", "avi", "mov", "mkv"])],
    )

    raw_stride = data.get("frame_stride", "auto")
    frame_stride: Union[int, Literal["auto"]]
    if isinstance(raw_stride, str) and raw_stride.lower() == "auto":
        frame_stride = "auto"
    else:
        frame_stride = int(raw_stride)

    config = EdgeVisionConfig(
        model=model_cfg,
        classes=classes_cfg,
        conf_threshold=float(data.get("conf_threshold", 0.20)),
        class_thresholds=dict(data.get("class_thresholds", {
            "person": 0.22,
            "bicycle": 0.15,
            "car": 0.12,
            "motorcycle": 0.15,
            "bus": 0.14,
            "truck": 0.14,
        })),
        nms_iou=float(data.get("nms_iou", 0.35)),
        inter_class_vehicle_nms=bool(data.get("inter_class_vehicle_nms", True)),
        inter_class_iou=float(data.get("inter_class_iou", 0.40)),
        max_detections=int(data.get("max_detections", 100)),
        frame_stride=frame_stride,
        threads=int(data.get("threads", 4)),
        smoothing=smoothing_cfg,
        zone=zone_cfg,
        congestion=congestion_cfg,
        privacy=privacy_cfg,
        output=output_cfg,
        upload=upload_cfg,
    )

    config.validate()
    return config
