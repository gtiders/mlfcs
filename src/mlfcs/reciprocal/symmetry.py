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
from mlfcs.reciprocal.grid import rotate_label
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
    "maximum_covariance_residual",
]
