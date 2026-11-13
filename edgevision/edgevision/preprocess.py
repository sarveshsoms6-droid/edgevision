"""Preprocessing routines for input frames.

Handles:
- Frame resizing to model input dimensions (e.g., 320x320) using cv2.INTER_LINEAR.
- Colour conversion (BGR to RGB).
- Normalisation and tensor float32 buffer preallocation to avoid per-frame allocations.
"""

from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """Preallocates buffers and transforms raw video frames for detector ingestion."""

    def __init__(self, input_size: int = 320) -> None:
        """Initialize preprocessor with target square input dimension."""
        self.input_size = input_size
        # Preallocated float32 tensor buffer [1, 3, H, W]
        self._buffer = np.zeros((1, 3, input_size, input_size), dtype=np.float32)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize, convert colour, normalise, and format frame into preallocated buffer.

        Args:
            frame: Raw BGR video frame from OpenCV (H, W, C).

        Returns:
            Preallocated float32 array with shape (1, 3, input_size, input_size)
            normalized to [-1.0, 1.0].
        """
        # Resize using INTER_LINEAR
        resized = cv2.resize(
            frame,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # SSDLite MobileNetV3 normalization: (x / 255.0 - 0.5) / 0.5 == x / 127.5 - 1.0
        # Transpose HWC (320, 320, 3) to CHW (3, 320, 320) directly into buffer
        norm = (rgb.astype(np.float32) / 127.5) - 1.0
        transposed = np.transpose(norm, (2, 0, 1))
        np.copyto(self._buffer[0], transposed)

        return self._buffer
