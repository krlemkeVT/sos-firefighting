"""Add local package sources to ``sys.path`` for package-level tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for relative_path in [
    "packages/triton_api/src",
    "packages/triton_io/src",
    "packages/triton_runner/src",
    "packages/optimizer/src",
]:
    source_path = str(ROOT / relative_path)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
