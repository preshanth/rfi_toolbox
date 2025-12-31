"""
Evaluation metrics for RFI detection.

Provides standard segmentation metrics (IoU, F1, Dice, Precision, Recall)
and statistical evaluation (FFI - Flagging Fidelity Index).
"""

from .metrics import (
    compute_iou,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_dice,
    evaluate_segmentation,
)
from .statistics import (
    compute_statistics,
    compute_ffi,
    print_statistics_comparison,
)

__all__ = [
    "compute_iou",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_dice",
    "evaluate_segmentation",
    "compute_statistics",
    "compute_ffi",
    "print_statistics_comparison",
]
