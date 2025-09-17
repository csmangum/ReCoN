from __future__ import annotations

from typing import Tuple


def normalize_signed(value: float, vmin: float, vmax: float) -> float:
    """Map signed value to [0,1] with 0.5 as zero midpoint.

    Assumes vmin < 0 < vmax; clamps out-of-range.
    """
    if vmax == vmin:
        return 0.5
    # Map to [-1,1] then to [0,1]
    span = max(abs(vmin), abs(vmax))
    x = max(-span, min(span, value)) / span
    return 0.5 * (x + 1.0)


def normalize_unsigned(value: float, vmin: float, vmax: float) -> float:
    if vmax == vmin:
        return 0.0
    x = (value - vmin) / (vmax - vmin)
    return max(0.0, min(1.0, float(x)))


def edge_width_from_flow(value: float, vmin: float, vmax: float, min_w: float = 1.0, max_w: float = 6.0) -> float:
    a = normalize_unsigned(value, max(0.0, vmin), max(vmin, vmax))
    return min_w + a * (max_w - min_w)

