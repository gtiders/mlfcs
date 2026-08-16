from __future__ import annotations


def zero_small_scalar(value: float, *, tolerance: float) -> float:
    """Return exact zero for text-export noise below an explicit tolerance."""
    return 0.0 if abs(value) < tolerance else float(value)


__all__ = ["zero_small_scalar"]
