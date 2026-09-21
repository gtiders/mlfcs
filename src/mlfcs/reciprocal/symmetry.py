r"""Space-group representation of the mass-weighted displacement space.

The dynamical matrix is not merely labelled by a q point: it transforms covariantly under
every symmetry operation that keeps the reciprocal grid, and any reduction to irreducible
points, any expansion of a covariance matrix and any symmetry-aware random sampling has to
use that relation.

Derivation
==========

With row-vector fractional coordinates the operation acts as
:math:`f' = f R^{\mathsf T} + t`, the site permutation is :math:`\pi` (``site_permutations``)
and the Cartesian rotation acting on row vectors is :math:`M` (``cartesian_rotations``), so
a displacement transforms as :math:`u'_{\pi(a)} = M^{\mathsf T} u_a` in the column
convention used for the 3-vectors of the mass-weighted space.  A plane wave
:math:`\exp[2 \pi i\, q \cdot (R + \tau_a)]` in the positional gauge maps the momentum to
:math:`gq = q R^{-1}`, so

.. math::

    \tilde u'_{\pi(a)}(gq) = \exp\left[-2 \pi i\, (gq) \cdot t\right] M^{\mathsf T}
        \tilde u_a(q),

which is exactly

.. math::

    D(gq) = U_g(q)\, D(q)\, U_g(q)^{\dagger}, \qquad
    U_g(q)_{\pi(a) a} = \exp\left[-2 \pi i\, (gq) \cdot t\right] M^{\mathsf T}.

The phase argument is formed from the *unreduced* integer label of :math:`gq` and the
operation translation, so it is an exact rational multiple of :math:`2 \pi` and does not
depend on how the label was rounded.  Reducing the label modulo the grid first would fold
in a primitive reciprocal lattice translation, which the positional gauge turns into
:math:`\Gamma_G = \operatorname{diag}(\exp(2 \pi i\, G \cdot \tau_a))`; the covariance
relation is therefore evaluated on the unreduced label, and the reduced action is reserved
for label sets such as stars and orbits.  For an antiunitary member (time reversal) the same
relation holds with a complex conjugation applied to :math:`D(q)`:
:math:`D(-q) = \overline{D(q)}`, which :func:`conjugate_matrix` states explicitly instead
of hiding it inside a "rotation".

The functions here only *measure* that covariance: they never average a violation away.
A force-constant set that breaks the symmetry of its structure is reported with the
operation index, the label and the residual, so the caller can decide what to do with it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from mlfcs.reciprocal.fourier import FourierTerm, dynamical_matrix
from mlfcs.reciprocal.grid import IrreducibleReciprocalGrid, rotate_label
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations


class RepresentationResidual(NamedTuple):
    """One measured covariant failure: ``operation`` and ``label`` with its residual."""

    operation: int
    label: tuple[int, int, int]
    residual: float


def displacement_representation(
    symmetry: PrimitiveSymmetryOperations,
    operation: int,
    label: np.ndarray,
    denominator: int,
) -> np.ndarray:
    r"""Return the unitary :math:`U_g(q)` of one grid-preserving operation at one q label.

    ``label`` are the reduced integer numerators of ``q`` (``q = label / denominator``),
    and the returned matrix acts on component vectors ordered
    ``[atom0 x, y, z, atom1 x, y, z, ...]`` of the mass-weighted displacement space.
    """
    if denominator < 1:
        raise ValueError("denominator must be positive")
    rotation = np.asarray(symmetry.rotations[operation], dtype=np.int64)
    translation = np.asarray(symmetry.translations[operation], dtype=float)
    permutation = np.asarray(symmetry.site_permutations[operation], dtype=np.int64)
    cartesian = np.asarray(symmetry.cartesian_rotations[operation], dtype=float)
    rotated = np.asarray(
        rotate_label(np.asarray(label, dtype=np.int64), rotation, denominator, reduce=False),
        dtype=np.int64,
    )
    phase = np.exp(-2j * np.pi * float(rotated @ translation) / denominator)
    n = len(permutation)
    unitary = np.zeros((3 * n, 3 * n), dtype=complex)
    block = phase * cartesian.T
    for site in range(n):
        target = int(permutation[site])
        unitary[3 * target : 3 * target + 3, 3 * site : 3 * site + 3] = block
    return unitary


def star_member_operator(
    symmetry: PrimitiveSymmetryOperations,
    grid: IrreducibleReciprocalGrid,
    member: int,
) -> tuple[np.ndarray, bool]:
    r"""Return the unitary carrying star data from a representative onto one member.

    ``member`` is a full-grid index; the star decomposition records, for it, the
    primitive operation that maps the representative of its star onto it, and whether
    that map also applies time reversal.  A quantity defined at the representative is
    therefore expanded by

    .. math::

        W(gq_s) = U_g\, W(q_s)\, U_g^{\dagger}, \qquad
        W(g(-q_s)) = U_g\, \overline{W(q_s)}\, U_g^{\dagger},

    and a *vector* by ``u'(gq) = U_g u(q)``, again conjugated first on an antiunitary
    member.  The phase of :math:`U_g` is one global scalar per operation, so it drops out
    of every similarity transform and only matters for vectors; a vector expansion must
    synthesize its plane waves in the gauge of :mod:`mlfcs.reciprocal.fourier`.
    """
    index = int(member)
    if index < 0 or index >= len(grid.full.labels):
        raise ValueError(f"member {index} is not a full-grid index of this decomposition")
    star = int(grid.full_to_irreducible[index])
    representative = int(grid.representatives[star])
    operation = int(grid.full_operations[index])
    unitary = displacement_representation(
        symmetry, operation, grid.full.labels[representative], grid.full.denominator
    )
    return unitary, bool(grid.full_antiunitary[index])


def expand_star_values(values: np.ndarray, grid: IrreducibleReciprocalGrid) -> np.ndarray:
    """Return the full-grid array of a quantity defined on star representatives.

    Star members carry the *same* eigenvalue data as their representative: the
    covariance relation is a unitary similarity, so frequencies and any other spectral
    quantity of ``D(q)`` are equal on the whole star.  Only the ordering inside a
    degenerate subspace may differ, which is why quantities compared across a star have
    to be spectral (sorted eigenvalues, projectors, covariance matrices).
    """
    array = np.asarray(values)
    if array.shape[:1] != (len(grid.representatives),):
        raise ValueError(
            f"expected one entry per representative ({len(grid.representatives)}), "
            f"got shape {array.shape}"
        )
    return array[grid.full_to_irreducible]


def expand_star_matrices(
    matrices: np.ndarray,
    grid: IrreducibleReciprocalGrid,
    symmetry: PrimitiveSymmetryOperations,
) -> np.ndarray:
    """Return the full-grid matrices ``W(gq_s)`` of per-representative matrices.

    This is the exact expansion of stage E: every member is reached by applying the
    recorded operation to the representative, with a complex conjugation first when the
    member is antiunitary.  It never averages and never multiplies a representative by a
    star weight.
    """
    values = np.asarray(matrices)
    n_irreducible = len(grid.representatives)
    if values.shape[:1] != (n_irreducible,):
        raise ValueError(
            f"expected one matrix per representative ({n_irreducible}), got shape {values.shape}"
        )
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError(f"expected square matrices, got shape {values.shape}")
    expanded = np.empty((len(grid.full.labels),) + values.shape[1:], dtype=values.dtype)
    for member in range(len(grid.full.labels)):
        star = int(grid.full_to_irreducible[member])
        unitary, antiunitary = star_member_operator(symmetry, grid, member)
        source = np.conjugate(values[star]) if antiunitary else values[star]
        expanded[member] = unitary @ source @ unitary.conj().T
    return expanded


def conjugate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the matrix of an antiunitary (time-reversal) member.

    For a real force-constant model the antiunitary partner of ``D(q)`` is
    :math:`\\overline{D(q)} = D(-q)`, which is an operation on the matrix itself and not a
    rotation of the displacement space.
    """
    return np.conjugate(matrix)


def covariance_residuals(
    terms: tuple[FourierTerm, ...],
    masses: np.ndarray,
    symmetry: PrimitiveSymmetryOperations,
    labels: np.ndarray,
    denominator: int,
    *,
    operations: np.ndarray | None = None,
) -> tuple[RepresentationResidual, ...]:
    """Measure ``||D(gq) - U_g(q) D(q) U_g(q)^dagger||`` for every operation and label.

    The q points are the reduced integer ``labels`` of the caller's grid.  Passing the
    grid-preserving ``operations`` restricts the measurement to those; the default is every
    operation of the primitive symmetry, which also exposes operations that do *not* keep
    the grid and are therefore not usable for an expansion.
    """
    values = np.asarray(labels, dtype=np.int64).reshape((-1, 3))
    indices = (
        np.arange(symmetry.size, dtype=np.int64)
        if operations is None
        else np.asarray(operations, dtype=np.int64)
    )
    residuals = []
    for index in indices:
        operation = int(index)
        rotation = np.asarray(symmetry.rotations[operation], dtype=np.int64)
        for row in values:
            # The reduction modulo the grid is withheld: it is a primitive reciprocal
            # lattice translation, i.e. a site-diagonal gauge factor in the positional
            # gauge of the dynamical matrix, not a change of the representation.
            rotated = np.asarray(
                rotate_label(row, rotation, denominator, reduce=False), dtype=float
            )
            left = dynamical_matrix(terms, masses, rotated / denominator)
            unitary = displacement_representation(symmetry, operation, row, denominator)
            right = unitary @ dynamical_matrix(terms, masses, row / denominator) @ unitary.conj().T
            residuals.append(
                RepresentationResidual(
                    operation,
                    tuple(int(value) for value in row),
                    float(np.max(np.abs(left - right))),
                )
            )
    return tuple(residuals)


def maximum_covariance_residual(
    terms: tuple[FourierTerm, ...],
    masses: np.ndarray,
    symmetry: PrimitiveSymmetryOperations,
    labels: np.ndarray,
    denominator: int,
    *,
    operations: np.ndarray | None = None,
) -> float:
    """Return the largest covariant residual over the requested operations and labels."""
    residuals = covariance_residuals(
        terms, masses, symmetry, labels, denominator, operations=operations
    )
    return max((residual.residual for residual in residuals), default=0.0)


__all__ = [
    "RepresentationResidual",
    "conjugate_matrix",
    "covariance_residuals",
    "displacement_representation",
    "expand_star_matrices",
    "expand_star_values",
    "maximum_covariance_residual",
    "star_member_operator",
]
