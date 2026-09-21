"""Compiled kernels for the reciprocal numerical plans.

The kernels take only flattened, fixed-dtype arrays from :mod:`mlfcs.reciprocal.plan`: no
dataclasses, no tuple-keyed dictionaries and no term objects, which is what makes them
compilable at all.  ``fastmath`` stays off, so the arithmetic is the same arithmetic the NumPy
oracle performs; the kernels change where the loop runs, not what it computes.

The Hermitian fold is done inside the kernel, which removes the intermediate non-Hermitian
matrix the pure NumPy path allocates.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from mlfcs.reciprocal.plan import FourierPlan

_TWO_PI = 2.0 * np.pi


@njit(cache=True, nogil=True)
def _dynamical_matrices_kernel(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    images: np.ndarray,
    tensors: np.ndarray,
    mass_weight: np.ndarray,
    n_sites: int,
    out: np.ndarray,
) -> None:
    """Accumulate every term into ``out`` and fold it in place.

    Loop order is fixed as q point, term, Cartesian block, so the accumulation order matches the
    term tuple the NumPy path walks; only the per-term overhead disappears.
    """
    n_terms = first.shape[0]
    n_points = points.shape[0]
    for iq in range(n_points):
        for term in range(n_terms):
            angle = 0.0
            for axis in range(3):
                angle += images[term, axis] * points[iq, axis]
            angle *= _TWO_PI
            weight = mass_weight[term]
            re = np.cos(angle) * weight
            im = np.sin(angle) * weight
            row = 3 * first[term]
            column = 3 * second[term]
            for a in range(3):
                for b in range(3):
                    value = tensors[term, a, b]
                    out[iq, row + a, column + b] += complex(value * re, value * im)
    for iq in range(n_points):
        for a in range(3 * n_sites):
            for b in range(a, 3 * n_sites):
                value = 0.5 * (out[iq, a, b] + np.conj(out[iq, b, a]))
                out[iq, a, b] = value
                out[iq, b, a] = np.conj(value)


def dynamical_matrices_compiled(plan: FourierPlan, qpoints: np.ndarray) -> np.ndarray:
    """Return the mass-weighted dynamical matrices of a q batch through the compiled kernel.

    The result is the same quantity as ``fourier.dynamical_matrices`` for the same terms, to
    roundoff; the caller decides which one to use.
    """
    points = np.asarray(qpoints, dtype=float).reshape((-1, 3))
    out = np.zeros((len(points), 3 * plan.n_sites, 3 * plan.n_sites), dtype=complex)
    _dynamical_matrices_kernel(
        np.ascontiguousarray(points),
        np.ascontiguousarray(plan.first),
        np.ascontiguousarray(plan.second),
        np.ascontiguousarray(plan.images),
        np.ascontiguousarray(plan.tensors),
        np.ascontiguousarray(plan.mass_weight),
        int(plan.n_sites),
        out,
    )
    return out


def kernel_is_nopython() -> bool:
    """Return whether the production dispatcher compiled to nopython mode.

    A kernel that fell back to object mode would be slower than the NumPy path it replaces, so
    the test suite asks this instead of assuming.
    """
    signatures = _dynamical_matrices_kernel.signatures
    if not signatures:
        return False
    return all(signature[0] != "object" for signature in signatures)


__all__ = ["dynamical_matrices_compiled", "kernel_is_nopython"]
