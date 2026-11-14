"""Vercel Serverless Function entry point for EdgeVision web application.
Routes incoming HTTP requests through Flask with Vercel serverless compatibility.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Add project root and edgevision folder to Python path
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
_edgevision_dir = _project_root / "edgevision"

for p in [str(_project_root), str(_edgevision_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Redirect writable directories to /tmp on serverless environments
if os.environ.get("VERCEL"):
    os.environ.setdefault("TMPDIR", "/tmp")

from app.server import app

# Vercel's @vercel/python builder looks for the 'app' WSGI object
__all__ = ["app"]
