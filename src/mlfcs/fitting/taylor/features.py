"""Ordinary Taylor monomials used as fitting coordinates."""

from __future__ import annotations

import numpy as np


def taylor_axis_derivatives(displacement, _state, coordinates, order):
    """Return the leave-one-axis monomial for every tensor axis.

    This feature is evaluated inside jitted design kernels.  ``np.prod`` and
    ``np.asarray`` are used only on a host-side axis index, so numpy dispatches
    the reduction to the array's own ``prod`` and the values stay traced; the
    monomial is never materialised on the host.
    """
    values = displacement.reshape(-1)[coordinates]
    return tuple(
        np.prod(
            values[..., np.asarray([other for other in range(order) if other != axis])],
            axis=-1,
        )
        for axis in range(order)
    )


__all__ = ["taylor_axis_derivatives"]
