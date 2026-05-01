from __future__ import annotations

"""
Dynamic fusion utilities are defined in src/fusion.py. This file provides a small
shim so callers can import from src.dynamic_fusion if desired without breaking
existing imports.
"""

from .fusion import (
    USE_NEW_FUSION,
    AudioPred,
    compare_fusions,
    compute_weights,
    fuse_predictions,
    fuse_predictions_new,
)

__all__ = [
    "USE_NEW_FUSION",
    "AudioPred",
    "compare_fusions",
    "compute_weights",
    "fuse_predictions",
    "fuse_predictions_new",
]
