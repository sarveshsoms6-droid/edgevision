"""Preprocessing routines for input frames.

Handles:
- Frame resizing to model input dimensions (e.g., 320x320) using cv2.INTER_LINEAR.
- Colour conversion (BGR to RGB).
- Normalisation based on model family:
  - 'ssd': [-1.0, 1.0] range via (pixel / 127.5) - 1.0
  - 'fasterrcnn': [0.0, 1.0] range via pixel / 255.0
- Preallocation of float32 buffer to avoid per-frame allocations.
"""

from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """Preallocates buffers and transforms raw video frames for detector ingestion."""

    def __init__(self, input_size: int = 320, norm_mode: str = "ssd") -> None:
        """Initialize preprocessor with target square input dimension and normalization mode.

        Args:
            input_size: Target square image dimension (e.g. 320).
            norm_mode: Normalization mode ('ssd' for [-1, 1], 'fasterrcnn' for [0, 1]).
        """
        self.input_size = input_size
        self.norm_mode = norm_mode
        # Preallocated float32 tensor buffer [1, 3, H, W]
        self._buffer = np.zeros((1, 3, input_size, input_size), dtype=np.float32)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize, convert colour, normalise, and format frame into preallocated buffer.

        Args:
            frame: Raw BGR video frame from OpenCV (H, W, C).

        Returns:
            Preallocated float32 array with shape (1, 3, input_size, input_size).
        """
        # Resize using INTER_LINEAR
        resized = cv2.resize(
            frame,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.norm_mode == "fasterrcnn":
            # Faster R-CNN expects standard [0, 1] input
            norm = rgb.astype(np.float32) / 255.0
        else:
            # SSDLite MobileNetV3: (x / 255.0 - 0.5) / 0.5 == x / 127.5 - 1.0
            norm = (rgb.astype(np.float32) / 127.5) - 1.0

        # Transpose HWC to CHW directly into preallocated buffer
        transposed = np.transpose(norm, (2, 0, 1))
        np.copyto(self._buffer[0], transposed)

        return self._buffer
