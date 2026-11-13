"""Video input/output and frame ingestion module.

Handles:
- Video file metadata extraction (resolution, FPS, total frames, duration).
- Duration and file format validation.
- Ingest downscaling for videos wider than 640px.
- Background decoder thread reading preprocessed frames into a bounded queue (size 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from typing import Generator, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Metadata container for ingested video files."""

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Extract resolution, FPS, total frame count, and duration from video file.

    Args:
        video_path: Path to target video file.

    Returns:
        VideoMetadata dataclass.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV cannot open or read the video file.
    """
    path_obj = Path(video_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0.0 or np.isnan(fps):
            fps = 30.0  # Safe fallback for variable framerate streams

        duration = (total_frames / fps) if total_frames > 0 else 0.0

        return VideoMetadata(
            path=str(path_obj),
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
        )
    finally:
        cap.release()


def validate_video_duration(video_path: str | Path, max_seconds: float = 60.0) -> bool:
    """Verify that video duration does not exceed the allowed threshold.

    Args:
        video_path: Path to target video file.
        max_seconds: Maximum permitted video duration in seconds.

    Returns:
        True if duration <= max_seconds, False otherwise.
    """
    meta = get_video_metadata(video_path)
    return meta.duration_seconds <= max_seconds


def read_video_frames(
    video_path: str | Path,
    max_width: int = 640,
    frame_stride: int = 1,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """Synchronous generator that iterates through video frames with downscaling and stride.

    Args:
        video_path: Path to video file.
        max_width: Maximum width allowed before downscaling (ingest downscale).
        frame_stride: Read every Nth frame.

    Yields:
        Tuple of (frame_index, timestamp_seconds, frame_bgr).
    """
    path_obj = Path(video_path)
    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0 or np.isnan(fps):
        fps = 30.0

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_stride == 0:
                h, w = frame.shape[:2]
                if w > max_width:
                    scale = max_width / float(w)
                    new_w = max_width
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                timestamp = frame_idx / fps
                yield frame_idx, timestamp, frame

            frame_idx += 1
    finally:
        cap.release()


class VideoDecoderThread:
    """Background decoder thread reading frames into a bounded queue (size 4)."""

    def __init__(
        self,
        video_path: str | Path,
        queue_size: int = 4,
        frame_stride: int = 1,
        max_width: int = 640,
    ) -> None:
        self.video_path = str(video_path)
        self.queue_size = queue_size
        self.frame_stride = frame_stride
        self.max_width = max_width
        self.queue: Queue[Optional[Tuple[int, float, np.ndarray]]] = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _worker(self) -> None:
        try:
            for item in read_video_frames(
                self.video_path,
                max_width=self.max_width,
                frame_stride=self.frame_stride,
            ):
                if self._stop_event.is_set():
                    break
                while not self._stop_event.is_set():
                    try:
                        self.queue.put(item, timeout=0.1)
                        break
                    except:
                        continue
        finally:
            # Signal completion with sentinel None
            while not self._stop_event.is_set():
                try:
                    self.queue.put(None, timeout=0.1)
                    break
                except:
                    continue

    def start(self) -> None:
        """Start the background reading thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def get_frame(self, timeout: float = 5.0) -> Optional[Tuple[int, float, np.ndarray]]:
        """Retrieve next frame from bounded queue. Returns None when video ends."""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        """Stop background decoding and release resources."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
