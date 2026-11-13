"""EdgeVision: Offline Vehicle and Pedestrian Detection & Counting System.

Designed for edge deployment on Raspberry Pi 4 using ExecuTorch and XNNPACK.
"""

from edgevision.config import EdgeVisionConfig, load_config

__version__ = "0.1.0"
__all__ = ["EdgeVisionConfig", "load_config", "__version__"]
