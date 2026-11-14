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
    """Extract resolution, FPS, total frame count, and duration from video or image file.

    Args:
        video_path: Path to target video or image file.

    Returns:
        VideoMetadata dataclass.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV cannot open or read the video file.
    """
    path_obj = Path(video_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    # Directly handle static image files
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if path_obj.suffix.lower() in image_exts:
        img = cv2.imread(str(path_obj))
        if img is None or img.size == 0:
            raise ValueError(f"Could not open or decode image file: {video_path}")
        h, w = img.shape[:2]
        return VideoMetadata(
            path=str(path_obj),
            width=w,
            height=h,
            fps=30.0,
            total_frames=1,
            duration_seconds=1.0 / 30.0,
        )

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle stream or missing dimensions by reading the first test frame
        if width <= 0 or height <= 0:
            ret, test_frame = cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                height, width = test_frame.shape[:2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if fps <= 0.0 or np.isnan(fps) or fps > 120.0:
            fps = 30.0  # Safe fallback for variable framerate streams

        if total_frames <= 0:
            # Estimate from duration or fallback to conservative count
            total_frames = 1

        duration = (total_frames / fps) if total_frames > 0 else 0.0

        return VideoMetadata(
            path=str(path_obj),
            width=max(1, width),
            height=max(1, height),
            fps=fps,
            total_frames=max(1, total_frames),
            duration_seconds=max(0.0, duration),
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
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if path_obj.suffix.lower() in image_exts:
        img = cv2.imread(str(path_obj))
        if img is None or img.size == 0:
            raise ValueError(f"Failed to open image file: {video_path}")
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_w = max_width
            new_h = int(round(h * scale))
        else:
            new_w, new_h = w, h
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        if (w, h) != (new_w, new_h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        yield 0, 0.0, img
        return

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0 or np.isnan(fps) or fps > 120.0:
        fps = 30.0

    frame_idx = 0
    consecutive_drop_count = 0
    max_drop_tolerance = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                consecutive_drop_count += 1
                if consecutive_drop_count >= max_drop_tolerance:
                    break
                frame_idx += 1
                continue

            consecutive_drop_count = 0

            if frame_idx % frame_stride == 0:
                h, w = frame.shape[:2]
                if w > max_width:
                    scale = max_width / float(w)
                    new_w = max_width
                    new_h = int(round(h * scale))
                else:
                    new_w, new_h = w, h

                # Guarantee even dimensions for video codecs and downstream processors
                new_w = new_w - (new_w % 2)
                new_h = new_h - (new_h % 2)

                if (w, h) != (new_w, new_h):
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
                    except Exception:
                        continue
        except Exception:
            pass
        finally:
            # Signal completion with sentinel None
            while not self._stop_event.is_set():
                try:
                    self.queue.put(None, timeout=0.1)
                    break
                except Exception:
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
        # Drain queue if worker is blocked on put
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
