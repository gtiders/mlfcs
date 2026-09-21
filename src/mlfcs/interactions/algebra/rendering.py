r"""One entry point that renders exact lattice orbit bases into Cartesian components.

An orbit owns two bases of the same invariant subspace and every consumer goes through
this module to relate them:

* ``exact_lattice_basis`` $B_{\mathbb Z}$: integer columns in the reduced lattice
  (scaled) frame, the only frame in which the symmetry algebra, the rank certificate and
  the invariant kernel are exact;
* ``cartesian_basis`` $Q$: an orthonormal basis of the Cartesian subspace
  $C = K_n B_{\mathbb Z}$, which is what fitting, finite-difference reconstruction,
  constraints and force-constant expansion consume.  The fixed-tensor order map
  $K_n = (A^\mathsf{T})^{\otimes n}$ is the frame of the reduced algebra cell $A$.

Keeping the conversion in one place is what lets the fitted parameters carry exactly one
coordinate meaning: they are the coefficients $\theta$ of $Q$, while the exact lattice
coefficients are $c = R^{-1}\theta$ with $C = QR$.  No caller has to guess whether a
column holds lattice components, Cartesian components, or pivot-normalized values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import qr

from mlfcs.interactions.algebra.actions import as_int64

# Two row sets whose volumes differ by less than this are the same choice.
_VOLUME_EPSILON = 1e-12

if TYPE_CHECKING:  # pragma: no cover - import cycle free type-only import
    from mlfcs.structure.lattice_frame import LatticeFrame


def tensor_order(components: int) -> int:
    """Return the tensor order of ``3**order`` component columns."""
    order = 0
    size = 1
    while size < components:
        size *= 3
        order += 1
    if size != components:
        raise ValueError(f"{components} is not a power of three tensor component count")
    return order


def cartesian_orbit_basis(exact_lattice_basis: np.ndarray, frame: LatticeFrame) -> np.ndarray:
    r"""Return the Cartesian components $C = K_n B_{\mathbb Z}$ of an exact basis."""
    exact = as_int64(np.asarray(exact_lattice_basis), context="exact lattice orbit basis")
    if exact.ndim != 2:
        raise ValueError(f"expected a 2D exact basis, got shape {exact.shape}")
    order = tensor_order(int(exact.shape[0]))
    return frame.tensor_frame(order) @ exact


def orthonormal_orbit_basis(cartesian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the reduced QR factorization $C = QR$ with a positive ``diag(R)``.

    ``Q`` has orthonormal columns and ``R`` is the invertible upper triangular transform
    between the exact lattice coefficients and the fitted Cartesian ones.  Flipping the
    signs to make the diagonal of ``R`` positive removes the only sign ambiguity of a QR
    factorization, so the basis is a deterministic function of the subspace.
    """
    values = np.asarray(cartesian, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D Cartesian basis, got shape {values.shape}")
    if values.shape[1] == 0:
        return np.empty((len(values), 0), dtype=float), np.empty((0, 0), dtype=float)
    orthonormal, transform = qr(values, mode="economic", check_finite=False)
    signs = np.where(np.diag(transform) < 0.0, -1.0, 1.0)
    return orthonormal * signs, transform * signs[:, None]


def select_observation_rows(orthonormal: np.ndarray, dimension: int) -> np.ndarray:
    """Return the component rows a finite-difference plan observes for one orbit.

    Rows are added greedily to maximize the volume of the selected block, that is the
    determinant of ``Q[rows] Q[rows].T``; the first row that attains the largest volume
    wins, so the choice is a deterministic function of the subspace.  A maximal-volume
    row set keeps the observation matrix well conditioned, which a rank-revealing QR need
    not: it only bounds the trailing block, and on an order-4 anharmonic space its rows
    amplified the finite-difference truncation error by 10 percent against the
    maximal-volume set.

    These rows are *observations*, not parameter pivots: the parameters are the
    coefficients of ``Q``, and reconstruction solves ``Q[rows] @ theta = y[rows]``.
    """
    values = np.asarray(orthonormal, dtype=float)
    if dimension < 0 or dimension > values.shape[1]:
        raise ValueError(f"cannot select {dimension} observation rows from {values.shape}")
    if dimension == 0:
        return np.empty(0, dtype=np.int32)
    selected: list[int] = []
    for _ in range(dimension):
        best_row = -1
        best_score = -np.inf
        for row in range(values.shape[0]):
            if row in selected:
                continue
            trial = values[np.asarray((*selected, row), dtype=np.int64)]
            singular = np.linalg.svd(trial, compute_uv=False)
            # A rank-deficient candidate has zero volume and is never the best choice.
            score = float(np.sum(np.log(singular))) if np.all(singular > 0.0) else -np.inf
            if score > best_score + _VOLUME_EPSILON:
                best_row, best_score = row, score
        if best_row < 0:
            raise RuntimeError(f"could not select {dimension} independent observation rows")
        selected.append(best_row)
    return np.sort(np.asarray(selected, dtype=np.int32))


def observation_matrix(orthonormal: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return ``Q[rows]``: the square system that maps parameters to observations."""
    values = np.asarray(orthonormal, dtype=float)
    indices = np.asarray(rows, dtype=np.int64)
    if indices.ndim != 1 or len(indices) != values.shape[1]:
        raise ValueError(f"expected {values.shape[1]} observation rows, got shape {indices.shape}")
    return values[indices]


def observation_condition(matrix: np.ndarray) -> float:
    """Return the 2-norm condition number of an observation matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.size == 0:
        return 1.0
    return float(np.linalg.cond(values, 2))


def render_representative_tensor(
    cartesian_basis: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    r"""Return the Cartesian representative tensor of one orbit.

    ``cartesian_basis`` is $Q$ and ``coefficients`` the fitted parameters $\theta$; the
    representative tensor is $Q\,\theta$ reshaped to the tensor axes.
    """
    basis = np.asarray(cartesian_basis, dtype=float)
    parameters = np.asarray(coefficients, dtype=float).reshape(-1)
    if basis.shape[1] != len(parameters):
        raise ValueError(
            f"basis has {basis.shape[1]} columns but {len(parameters)} parameters were given"
        )
    return (basis @ parameters).reshape((3,) * tensor_order(int(basis.shape[0])))


def exact_lattice_coefficients(
    coefficient_transform: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    r"""Return the exact lattice coefficients $c$ of the Cartesian parameters $\theta$.

    With $C = QR$ and a representative tensor $C c = Q\theta$, the exact lattice
    coefficients are $c = R^{-1}\theta$.  They are the frame-independent provenance of a
    fitted tensor: $c$ describes the same tensor in the reduced lattice frame, where the
    integer symmetry algebra acted.
    """
    transform = np.asarray(coefficient_transform, dtype=float)
    values = np.asarray(parameters, dtype=float).reshape(-1)
    if transform.shape != (len(values), len(values)):
        raise ValueError(
            f"coefficient transform has shape {transform.shape} for {len(values)} parameters"
        )
    return np.linalg.solve(transform, values)


__all__ = [
    "cartesian_orbit_basis",
    "exact_lattice_coefficients",
    "observation_condition",
    "observation_matrix",
    "orthonormal_orbit_basis",
    "render_representative_tensor",
    "select_observation_rows",
    "tensor_order",
]
