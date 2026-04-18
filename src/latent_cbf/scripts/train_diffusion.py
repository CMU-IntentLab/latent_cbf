# Copyright (c) Sudeep Dasari, 2023

"""Delegates to diffusion4robotics/train.py (dubins diffusion BC)."""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_D4R = os.path.join(_REPO_ROOT, "diffusion4robotics")
if _D4R not in sys.path:
    sys.path.insert(0, _D4R)

from train import main  # noqa: E402

if __name__ == "__main__":
    main()
