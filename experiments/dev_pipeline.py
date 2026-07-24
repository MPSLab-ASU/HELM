"""Compatibility entrypoint for the packaged compiler diagnostic pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helm import _pipeline

if __name__ == "__main__":
    _pipeline.main()
else:
    # Keep imports and monkeypatches in existing research scripts compatible.
    sys.modules[__name__] = _pipeline
