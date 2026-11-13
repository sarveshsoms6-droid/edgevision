"""ExecuTorch runtime wrapper for edge inference using XNNPACK backend.

Loads compiled .pte models (FP32 or INT8), sets XNNPACK thread count,
manages preallocated memory buffers, runs warm-up iterations, and provides
high-resolution timing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch

from executorch.runtime import Program, Runtime


class ExecuTorchDetector:
    """ExecuTorch runtime detector wrapper with XNNPACK backend."""

    def __init__(
        self,
        pte_path: str | Path,
        runtime_mode: str = "executorch-fp32",
        threads: int = 4,
    ) -> None:
        """Initialize ExecuTorch detector runtime.

        Args:
            pte_path: Path to compiled .pte file.
            runtime_mode: Either 'executorch-fp32' or 'executorch-int8'.
            threads: Number of XNNPACK execution threads.

        Raises:
            FileNotFoundError: If the .pte model file does not exist.
        """
        p = Path(pte_path)
        if not p.is_file():
            # Search candidate paths relative to workspace or package
            candidates = [
                Path("models") / p.name,
                Path("edgevision/models") / p.name,
                Path(__file__).resolve().parent.parent / "models" / p.name,
            ]
            for cand in candidates:
                if cand.is_file():
                    p = cand
                    break
        if not p.is_file():
            raise FileNotFoundError(
                f"ExecuTorch model file not found at: {pte_path}. "
                "Ensure Stage 2 export_model.py (for fp32) or Stage 6 quantize_model.py (for int8) has run."
            )

        self.pte_path = p
        self.runtime_mode = runtime_mode
        self.threads = max(1, threads)
        self.runtime_name = runtime_mode

        # Configure CPU and XNNPACK threads
        torch.set_num_threads(self.threads)
        self.runtime = Runtime.get()
        try:
            self.runtime.backend_registry.set_option("XnnpackBackend", {"num_threads": self.threads})
        except Exception:
            pass

        # Load program and forward method
        self.program: Program = self.runtime.load_program(str(self.pte_path))
        self.forward_method = self.program.load_method("forward")

    def warmup(self, num_iterations: int = 5, input_size: int = 320) -> None:
        """Execute dummy inferences to warm up the runtime and CPU caches.

        Args:
            num_iterations: Number of warm-up iterations (default 5).
            input_size: Model input dimension.
        """
        dummy_tensor = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
        for _ in range(max(1, num_iterations)):
            self.forward_method.execute([dummy_tensor])

    def predict_raw(
        self, input_data: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute inference on input array and return raw model outputs.

        Args:
            input_data: Preprocessed float32 array or tensor [1, 3, H, W] in range [-1, 1].

        Returns:
            Tuple of raw (class_logits [N, num_classes], box_regression [N, 4]) NumPy arrays.
        """
        if isinstance(input_data, np.ndarray):
            tensor = torch.from_numpy(input_data).float()
        else:
            tensor = input_data.float()

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        outputs = self.forward_method.execute([tensor])
        logits = outputs[0][0].numpy()
        regressions = outputs[1][0].numpy()

        return logits, regressions
