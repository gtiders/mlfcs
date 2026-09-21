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

#: Every candidate row set whose log volume is within this of the best so far counts as a
#: tie, so the smallest row index wins; compare with the relative volume it describes.
_VOLUME_TOLERANCE = 1e-12

#: Largest accepted ``max(abs(Q.T @ Q - I))`` for an observation basis.  The basis comes
#: from a QR factorization and is orthonormal to machine precision (about ``1e-15`` on the
#: orbits of this branch); anything looser than this is a caller passing a non-orthonormal
#: basis, usually the raw Cartesian basis ``C``, whose volumes mean nothing.
_ORTHONORMAL_TOLERANCE = 1e-8

#: Largest accepted 2-norm condition number of the selected observation block.  A solve
#: loses about one decimal digit per decade of condition number, so ``1e6`` keeps ten of
#: the sixteen double-precision digits.  This is a bound on the sloppiness the selection
#: may absorb, not a state an orthonormal basis reaches: the greedy residual at step ``j``
#: is at least ``sqrt((d - j + 1) / rows)``, and no material of this branch passes 4.9.
_MAX_OBSERVATION_CONDITION = 1e6

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


def _require_orthonormal(basis: np.ndarray) -> None:
    """Reject a basis whose columns are not orthonormal to the documented tolerance."""
    columns = basis.shape[1]
    if columns == 0:
        return
    deviation = float(np.max(np.abs(basis.T @ basis - np.eye(columns))))
    if deviation > _ORTHONORMAL_TOLERANCE:
        raise ValueError(
            f"observation rows require an orthonormal basis: max(abs(Q.T @ Q - I)) = "
            f"{deviation:.3e} exceeds the accepted {_ORTHONORMAL_TOLERANCE:.0e}"
        )


def select_observation_rows(orthonormal: np.ndarray, dimension: int) -> np.ndarray:
    r"""Return the component rows a finite-difference plan observes for one orbit.

    ``orthonormal`` is $Q$, an orthonormal basis of the orbit's Cartesian subspace, with
    one row per tensor component and one column per parameter: **row** ``r`` is the
    component whose axes are ``np.unravel_index(r, (3,) * order)``, i.e. the rows are
    enumerated in the Cartesian component order of ``np.ndindex``, which is the order the
    finite-difference plans and the reconstruction use.  The basis must be orthonormal --
    ``max(abs(Q.T @ Q - I)) <= 1e-8`` -- because the selection maximizes volumes, which are
    only meaningful for an orthonormal basis; a larger deviation raises ``ValueError``
    naming it, and the usual cause is the raw Cartesian basis $C$ instead of its QR factor.

    Rows are added greedily to maximize the volume of the selected block: the Gram volume
    $\sqrt{\det(Q[S] Q[S]^T)}$, which for a square block is $|\det Q[S]|$ and always the
    product of the singular values.  Candidates are visited in ascending row order and
    replace the incumbent only when their log volume exceeds it by more than ``1e-12``, so
    a volume tie inside that tolerance resolves to the *smallest* row index, and the
    returned rows are ascending.  This is a greedy heuristic and *not* a global
    maximum-volume solution: it never compares against row sets it did not walk through.
    It is nevertheless a strong one -- and the reason to keep it -- because a
    maximal-volume row set keeps the observation matrix well conditioned, which a
    rank-revealing QR need not: it only bounds the trailing block, and on an order-4
    anharmonic space its rows amplified the finite-difference truncation error by
    10 percent against the maximal-volume set.

    Every candidate volume is unchanged by a right multiplication with an orthogonal
    matrix, $\det(Q[S] O O^T Q[S]^T) = \det(Q[S] Q[S]^T)$, so the selection is invariant
    under $Q \to Q O$: it is a function of the subspace alone, not of which orthonormal
    basis of it happens to be stored.  A selection whose 2-norm condition number exceeds
    :data:`_MAX_OBSERVATION_CONDITION` cannot be solved stably and raises ``RuntimeError``
    naming the value.

    These rows are *observations*, not parameter pivots: the parameters are the
    coefficients of ``Q``, and reconstruction solves ``Q[rows] @ theta = y[rows]``.
    """
    values = np.asarray(orthonormal, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected a 2D orthonormal basis, got shape {values.shape}")
    _require_orthonormal(values)
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
            if score > best_score + _VOLUME_TOLERANCE:
                best_row, best_score = row, score
        if best_row < 0:
            raise RuntimeError(f"could not select {dimension} independent observation rows")
        selected.append(best_row)
    rows = np.sort(np.asarray(selected, dtype=np.int32))
    condition = observation_condition(values[rows])
    if condition > _MAX_OBSERVATION_CONDITION:
        raise RuntimeError(
            f"the greedy observation rows have 2-norm condition number {condition:.6g}, "
            f"above the accepted maximum {_MAX_OBSERVATION_CONDITION:.0e}: the selected "
            "block cannot be solved for the parameters stably"
        )
    return rows


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
