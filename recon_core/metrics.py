"""
Metrics utilities for Request Confirmation Networks (Day 6).

This module provides:
- Binary precision/recall computation helpers
- Convenience helpers to read timing/efficiency metrics from the Engine

The Engine records the following statistics in `engine.stats`:
- terminal_request_count: total number of SUR requests issued to TERMINAL units
- terminal_request_counts_by_id: per-terminal request counts
- first_request_step: first time step a unit entered REQUESTED
- first_active_step: first time step a unit entered ACTIVE
- first_true_step: first time step a TERMINAL entered TRUE
- first_confirm_step: first time step a SCRIPT entered CONFIRMED
"""

from __future__ import annotations
from typing import Iterable, Dict, Any, Tuple


def binary_precision_recall(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """
    Compute precision and recall for binary labels.

    Args:
        y_true: Iterable of ground-truth binary labels {0,1}
        y_pred: Iterable of predicted binary labels {0,1}

    Returns:
        dict with keys: precision, recall, tp, fp, tn, fn
    """
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 0:
            tn += 1
        elif p == 0 and t == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def steps_to_first_confirm(engine, unit_id: str) -> int | None:
    """Return the first time step when the unit became CONFIRMED, or None."""
    return engine.stats.get("first_confirm_step", {}).get(unit_id)


def steps_to_first_true(engine, unit_id: str) -> int | None:
    """Return the first time step when the terminal became TRUE, or None."""
    return engine.stats.get("first_true_step", {}).get(unit_id)


def total_terminal_requests(engine) -> int:
    """Return the total number of SUR requests sent to TERMINAL units."""
    return int(engine.stats.get("terminal_request_count", 0))


def terminal_request_counts_by_id(engine) -> Dict[str, int]:
    """Return per-terminal request counts as a dict of unit_id -> count."""
    counts = engine.stats.get("terminal_request_counts_by_id", {})
    # Normalize to int values in case of JSON casts
    return {k: int(v) for k, v in counts.items()}

