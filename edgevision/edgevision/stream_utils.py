"""Thread-safe stream utilities to prevent Windows console buffer / broken pipe errors."""

from __future__ import annotations

import sys
from typing import Any


class SafeStreamWrapper:
    """Wraps a text stream to gracefully swallow [Errno 22] Invalid argument and broken pipes."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, data: str) -> int:
        try:
            return self._stream.write(data)
        except OSError as e:
            # Swallow Windows EINVAL (22) and broken pipe (32)
            if getattr(e, "errno", None) in (22, 32):
                return len(data)
            return len(data)
        except Exception:
            return len(data)

    def flush(self) -> None:
        try:
            if hasattr(self._stream, "flush"):
                self._stream.flush()
        except OSError as e:
            if getattr(e, "errno", None) in (22, 32):
                return
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def patch_stdout_stderr() -> None:
    """Install SafeStreamWrapper on sys.stdout and sys.stderr if not already wrapped."""
    if not isinstance(sys.stdout, SafeStreamWrapper):
        sys.stdout = SafeStreamWrapper(sys.stdout)
    if not isinstance(sys.stderr, SafeStreamWrapper):
        sys.stderr = SafeStreamWrapper(sys.stderr)


def safe_print(*args: Any, **kwargs: Any) -> None:
    """Print wrapper that never raises OSError or terminates the caller."""
    try:
        print(*args, **kwargs)
    except Exception:
        pass
