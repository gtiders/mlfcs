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

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from mlfcs.exceptions import SymmetryViolationError
from mlfcs.reciprocal.fourier import FourierTerm, dynamical_matrix
from mlfcs.reciprocal.grid import IrreducibleReciprocalGrid, rotate_label, rotate_labels
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
    r"""Return the *un-gauged* unitary that carries star data onto one member.

    ``member`` is a full-grid index; the star decomposition records, for it, the primitive
    operation that maps the representative of its star onto it, and whether that map also
    applies time reversal.  This is the operation alone: it lands on the *unreduced* image
    ``g q_s``, which differs from the member's stored label by a primitive reciprocal lattice
    translation.  Expansion must therefore go through :func:`star_member_action` or
    :func:`expand_star_matrices`, which apply the positional gauge ``Gamma_G`` as well; the
    bare operator is kept because a few diagnostics need the two pieces separately, and its
    name says which piece it is.
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


def star_member_gauge(
    symmetry: PrimitiveSymmetryOperations,
    grid: IrreducibleReciprocalGrid,
    member: int,
    positions: np.ndarray,
) -> np.ndarray:
    r"""Return the site-diagonal gauge that brings an expanded matrix onto its member.

    :func:`star_member_operator` carries ``D`` from a representative to the *unreduced*
    image :math:`g q_s`, which differs from the member's stored label by a primitive
    reciprocal lattice translation :math:`G`.  The positional gauge of the dynamical
    matrix turns that translation into the diagonal factor

    .. math::

        \Gamma_G = \operatorname{diag}\left(\exp\left[2\pi i\, G \cdot \tau_a
            \right]\right),

    so a matrix expanded from the representative has to be conjugated as
    :math:`\Gamma_G\, U_g\, W(q_s)\, U_g^{\dagger}\, \Gamma_G^{\dagger}` before it
    can be multiplied by the member's own Fourier phase.  Skipping it makes an expansion
    look wrong by a factor of order one, exactly on the members whose label reduction
    actually bites -- which is most members of a multi-atom cell and none of the ones a
    single-atom test happens to visit.

    ``positions`` are the primitive scaled positions ``\tau_a``, one row per site; the
    returned vector has one entry per Cartesian coordinate, so it multiplies a matrix of the
    ``3n``-dimensional mass-weighted displacement space directly.
    """
    index = int(member)
    if index < 0 or index >= len(grid.full.labels):
        raise ValueError(f"member {index} is not a full-grid index of this decomposition")
    star = int(grid.full_to_irreducible[index])
    representative = int(grid.representatives[star])
    label = np.asarray(grid.full.labels[representative], dtype=np.int64)
    rotation = np.asarray(symmetry.rotations[int(grid.full_operations[index])], dtype=np.int64)
    image = rotate_labels(label, rotation, grid.full.denominator, reduce=False)
    if bool(grid.full_antiunitary[index]):
        image = -image
    difference = (
        np.asarray(image, dtype=float) - np.asarray(grid.full.labels[index], dtype=float)
    ) / grid.full.denominator
    sites = np.asarray(positions, dtype=float) @ difference
    # One entry per Cartesian coordinate, so the result can be applied directly to a
    # matrix of the 3n-dimensional mass-weighted displacement space.
    return np.kron(np.exp(2j * np.pi * sites), np.ones(3, dtype=complex))


@dataclass(frozen=True, slots=True)
class StarMemberAction:
    r"""The complete action that carries star data from a representative onto one member.

    Expansion is ``W_m = Gamma_m U_m \overline{W_s}^{[a_m]} U_m^\dagger Gamma_m^\dagger``
    for a matrix and ``w_m = Gamma_m U_m \overline{w_s}^{[a_m]}`` for a vector, with the
    complex conjugation applied first on an antiunitary member.  Both shapes go through this
    one object, so the phase order is written once: :func:`star_member_operator` and
    :func:`star_member_gauge` are the two pieces it is built from, and no caller has to
    recombine them by hand.
    """

    member: int
    #: Full-grid index of the star representative this member is expanded from.
    representative: int
    #: Star index, i.e. the row of a per-representative array that belongs to this member.
    star: int
    operation: int
    antiunitary: bool
    unitary: np.ndarray
    gauge: np.ndarray

    def apply_to_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Return the member matrix of one representative matrix."""
        source = np.conjugate(matrix) if self.antiunitary else matrix
        expanded = self.unitary @ source @ self.unitary.conj().T
        return self.gauge[:, None] * expanded * self.gauge.conj()[None, :]

    def apply_to_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Return the member basis of one representative basis, applied column by column."""
        source = np.conjugate(vectors) if self.antiunitary else vectors
        return self.gauge[:, None] * (self.unitary @ source)


def star_member_action(
    symmetry: PrimitiveSymmetryOperations,
    grid: IrreducibleReciprocalGrid,
    member: int,
    primitive_positions: np.ndarray,
) -> StarMemberAction:
    """Return the composite member action of one full-grid point.

    This is the single entry point every expansion uses: the operation, the antiunitary flag
    and the positional gauge are gathered here so that the covariance, the sampler and the
    validation kernels cannot drift apart in phase order.
    """
    unitary, antiunitary = star_member_operator(symmetry, grid, member)
    gauge = star_member_gauge(symmetry, grid, member, primitive_positions)
    index = int(member)
    star = int(grid.full_to_irreducible[index])
    return StarMemberAction(
        member=index,
        representative=int(grid.representatives[star]),
        star=star,
        operation=int(grid.full_operations[index]),
        antiunitary=antiunitary,
        unitary=unitary,
        gauge=gauge,
    )


def validate_symprec(value: object, *, context: str) -> float:
    """Return a usable geometric tolerance, rejecting NaN, infinities and non-positive values.

    ``symprec`` decides which operations the structure has; a NaN or a non-positive value
    would silently produce a meaningless operation set, so it fails here instead.
    """
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{context}: symprec must be finite and positive, got {value!r}")
    return number


def validate_symmetry_tolerance(value: object, *, context: str) -> float | None:
    """Return a usable physical covariance tolerance, or ``None`` for the explicit opt-out.

    The check is never off by default, so the only way to disable it is an explicit ``None``;
    NaN and infinities are rejected because a comparison against NaN is always false and would
    therefore switch the gate off by accident.
    """
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(
            f"{context}: symmetry_tolerance must be None or finite and non-negative, "
            f"got {value!r}"
        )
    return number


@dataclass(frozen=True, slots=True)
class StarCovarianceResidual:
    """One member whose directly built matrix disagrees with its star expansion."""

    representative: tuple[int, int, int]
    member: tuple[int, int, int]
    operation: int
    antiunitary: bool
    residual: float


def _star_expansion(build, symmetry, grid, primitive_positions, *, plan=None):
    """Return ``(direct, expanded)`` full-grid matrices of one star expansion."""
    representatives = np.asarray(grid.representatives, dtype=np.int64)
    direct = build(grid.full.points)
    expanded = expand_star_matrices(
        build(grid.full.points[representatives]),
        grid,
        symmetry,
        primitive_positions,
        plan=plan,
    )
    return direct, expanded


def _star_residuals(direct, expanded, grid) -> tuple[StarCovarianceResidual, ...]:
    """Return one residual per full-grid member of an expansion."""
    residuals = []
    for member in range(len(grid.full.labels)):
        star = int(grid.full_to_irreducible[member])
        representative = int(grid.representatives[star])
        residuals.append(
            StarCovarianceResidual(
                representative=tuple(int(value) for value in grid.full.labels[representative]),
                member=tuple(int(value) for value in grid.full.labels[member]),
                operation=int(grid.full_operations[member]),
                antiunitary=bool(grid.full_antiunitary[member]),
                residual=float(np.max(np.abs(expanded[member] - direct[member]))),
            )
        )
    return tuple(residuals)


def star_covariance_residuals(
    build,
    symmetry: PrimitiveSymmetryOperations,
    grid: IrreducibleReciprocalGrid,
    primitive_positions: np.ndarray,
) -> tuple[StarCovarianceResidual, ...]:
    """Compare every expanded star member with the matrix built directly at its own label.

    This is the certificate a star expansion actually needs: the little group only fixes the
    representatives, so a model can satisfy it and still be wrong on the members.  ``build``
    maps an ``(n, 3)`` array of q points to the matching stack of dynamical matrices, so the
    whole measurement is two batched builds -- the representatives, and every full-grid point.
    """
    direct, expanded = _star_expansion(build, symmetry, grid, primitive_positions)
    return _star_residuals(direct, expanded, grid)


def require_star_covariance(
    build,
    symmetry: PrimitiveSymmetryOperations,
    grid: IrreducibleReciprocalGrid,
    primitive_positions: np.ndarray,
    *,
    tolerance: float | None,
    context: str = "the model",
    plan: object | None = None,
) -> float:
    r"""Raise unless every expanded star member matches the matrix built at its own label.

    The allowed residual is relative to the largest infinity norm over the members, taken over
    both the direct and the expanded matrices, because a legitimate model's absolute residual
    scales with the model itself.  The check never averages, projects or repairs the input.
    """
    from mlfcs.reciprocal.modes import require_finite

    direct, expanded = _star_expansion(build, symmetry, grid, primitive_positions, plan=plan)
    # Finiteness is checked before any comparison and before the tolerance is consulted: a NaN
    # would make every comparison false, so switching the tolerance off must not admit it.
    require_finite(direct, role="direct dynamical matrices", context=context)
    require_finite(expanded, role="expanded dynamical matrices", context=context)
    residuals = _star_residuals(direct, expanded, grid)
    if not residuals:
        return 0.0
    if tolerance is None:
        return float(max(entry.residual for entry in residuals))
    scale = float(
        max(
            np.linalg.norm(direct, ord=np.inf, axis=(-2, -1)).max(),
            np.linalg.norm(expanded, ord=np.inf, axis=(-2, -1)).max(),
        )
    )
    worst = max(residuals, key=lambda entry: entry.residual)
    allowed = tolerance * scale if scale > 0 else tolerance
    if not np.isfinite(scale) or not np.isfinite(worst.residual) or not np.isfinite(allowed):
        raise SymmetryViolationError(
            f"{context}: the covariance comparison is not finite (scale {scale!r}, residual "
            f"{worst.residual!r}, allowed {allowed!r})"
        )
    if worst.residual > allowed:
        raise SymmetryViolationError(
            f"{context}: the force constants are not covariant on the full star of "
            f"representative {list(worst.representative)}: member {list(worst.member)} reached "
            f"by operation {worst.operation}"
            f"{' (antiunitary)' if worst.antiunitary else ''} leaves a residual of "
            f"{worst.residual:.6e} against an allowed {allowed:.6e} "
            f"(scale {scale:.6e}, relative tolerance {tolerance:.3e}). The star expansion "
            "would mix inequivalent points; fix the force constants or raise "
            "symmetry_tolerance explicitly."
        )
    return float(worst.residual)


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
    representative_matrices: np.ndarray,
    grid: IrreducibleReciprocalGrid,
    symmetry: PrimitiveSymmetryOperations,
    primitive_positions: np.ndarray,
    *,
    plan: object | None = None,
) -> np.ndarray:
    r"""Expand representative matrices onto every full-grid member, gauge included.

    Each member is reached by the recorded operation with a complex conjugation first on an
    antiunitary member, and then carried onto *its own stored label* by the positional gauge
    ``Gamma_G``:

    .. math::

        W_m = \Gamma_m U_m \overline{W_s}^{[a_m]} U_m^{\dagger} \Gamma_m^{\dagger}.

    Leaving the gauge out lands the matrix on the unreduced image ``g q_s`` instead, which is
    wrong by a factor of order one on exactly the members whose label is reduced -- a
    single-atom cell or a coarse mesh never shows it.  There is deliberately no switch to skip
    the gauge: the un-gauged operation is available as :func:`star_member_operator` under a
    name that says what it is.
    """
    values = np.asarray(representative_matrices)
    n_irreducible = len(grid.representatives)
    n_primitive = len(primitive_positions)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError(f"expected a stack of square matrices, got shape {values.shape}")
    if values.shape[:1] != (n_irreducible,):
        raise ValueError(
            f"expected one matrix per representative ({n_irreducible}), got shape {values.shape}"
        )
    if values.shape[1] != 3 * n_primitive:
        raise ValueError(
            f"expected {3 * n_primitive} rows for {n_primitive} sites, got {values.shape[1]}"
        )
    positions = np.asarray(primitive_positions, dtype=float)
    if positions.shape != (n_primitive, 3):
        raise ValueError(f"expected positions of shape ({n_primitive}, 3), got {positions.shape}")
    if values.size == 0:
        raise ValueError("no representative matrices were supplied")
    dtype = np.result_type(values.dtype, np.complex128)
    expanded = np.empty((len(grid.full.labels),) + values.shape[1:], dtype=dtype)
    if plan is not None:
        # A shared plan has already computed the integer inverse, the permutation and the
        # gauge of every member; assembling from it is bit-identical to the uncached path.
        for member in range(len(grid.full.labels)):
            expanded[member] = plan.apply_to_matrix(member, values[plan.star(member)])
        return expanded
    for member in range(len(grid.full.labels)):
        action = star_member_action(symmetry, grid, member, positions)
        expanded[member] = action.apply_to_matrix(values[action.star])
    return expanded


def conjugate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the matrix of an antiunitary (time-reversal) member.

    For a real force-constant model the antiunitary partner of ``D(q)`` is
    :math:`\\overline{D(q)} = D(-q)`, which is an operation on the matrix itself and not a
    rotation of the displacement space.
    """
    return np.conjugate(matrix)


def validate_site_masses(
    symmetry: PrimitiveSymmetryOperations, masses: np.ndarray, *, context: str = "the model"
) -> None:
    """Raise unless every site permutation maps equal masses onto equal masses.

    A space group operation permutes atoms of the same species, so a mass may only move
    along its orbit.  A permutation that maps a heavy site onto a light one is not a
    symmetry of the mass-weighted problem at all, and the expansion would silently mix two
    different mass scales.
    """
    values = np.asarray(masses, dtype=float)
    permutations = np.asarray(symmetry.site_permutations, dtype=np.int64)
    if permutations.ndim != 2 or permutations.shape[1] != values.size:
        raise SymmetryViolationError(
            f"{context}: site permutations {permutations.shape} do not match {values.size} sites"
        )
    for operation, permutation in enumerate(permutations):
        moved = values[permutation]
        if not np.allclose(moved, values, rtol=0.0, atol=0.0):
            bad = int(np.flatnonzero(moved != values)[0])
            raise SymmetryViolationError(
                f"{context}: operation {operation} maps site {bad} with mass {values[bad]!r} "
                f"onto a site with mass {moved[bad]!r}, so the mass-weighted representation "
                "of that operation is not defined"
            )


def _residuals_at(
    build,
    masses: np.ndarray,
    symmetry: PrimitiveSymmetryOperations,
    labels: np.ndarray,
    denominator: int,
    operations: np.ndarray | None,
) -> tuple[RepresentationResidual, ...]:
    """Measure the covariant residual of a single-q matrix builder over given operations."""
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
            left = build(rotated / denominator)
            unitary = displacement_representation(symmetry, operation, row, denominator)
            right = unitary @ build(row / denominator) @ unitary.conj().T
            residuals.append(
                RepresentationResidual(
                    operation,
                    tuple(int(value) for value in row),
                    float(np.max(np.abs(left - right))),
                )
            )
    return tuple(residuals)


def little_group_residuals(
    build,
    masses: np.ndarray,
    symmetry: PrimitiveSymmetryOperations,
    grid: IrreducibleReciprocalGrid,
) -> tuple[RepresentationResidual, ...]:
    """Measure the un-gauged covariance residual on the little group of every representative.

    This is a *local* diagnostic, not a star certificate: the little group fixes the
    representative, so a model can satisfy this measurement and still be wrong on every other
    member of the star -- the merge gate for an expansion is
    :func:`require_star_covariance`, which compares the expanded members with the matrices
    built at their own labels.  What is measured here is the un-gauged relation
    ``D(gq_s) = U_g(q_s) D(q_s) U_g(q_s)^dagger`` at the *unreduced* image, which is the
    relation :func:`covariance_residuals` and the representation tests use.
    """
    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for star, little_group in enumerate(grid.little_groups):
        label = np.asarray(grid.full.labels[int(grid.representatives[star])], dtype=np.int64)
        for operation in np.asarray(little_group, dtype=np.int64).tolist():
            rotation = np.asarray(symmetry.rotations[int(operation)], dtype=np.int64)
            # The reduction modulo the grid is withheld: it is a primitive reciprocal
            # lattice translation, i.e. a site-diagonal gauge factor in the positional
            # gauge of the dynamical matrix, not a change of the representation.
            rotated = rotate_labels(label, rotation, grid.full.denominator, reduce=False)
            pairs.append((int(operation), label, np.asarray(rotated, dtype=float)))
    if not pairs:
        return ()
    denominator = grid.full.denominator
    targets = build(np.asarray([pair[2] for pair in pairs]) / denominator)
    sources = build(np.asarray([pair[1] for pair in pairs], dtype=float) / denominator)
    residuals = []
    for index, (operation, label, _) in enumerate(pairs):
        unitary = displacement_representation(symmetry, operation, label, denominator)
        residuals.append(
            RepresentationResidual(
                operation,
                tuple(int(value) for value in label),
                float(
                    np.max(
                        np.abs(
                            targets[index] - unitary @ sources[index] @ unitary.conj().T
                        )
                    )
                ),
            )
        )
    return tuple(residuals)


def require_little_group_covariance(
    build,
    masses: np.ndarray,
    symmetry: PrimitiveSymmetryOperations,
    grid: IrreducibleReciprocalGrid,
    *,
    tolerance: float | None,
    context: str = "the model",
) -> float:
    """Raise unless the un-gauged little-group residual holds within a relative tolerance.

    A local diagnostic only; use :func:`require_star_covariance` as the gate for any
    expansion over a star.  ``build`` maps an ``(n, 3)`` array of q points to the matching
    stack of dynamical matrices, and ``tolerance`` is relative to the largest
    dynamical-matrix element at the representatives.
    """
    if tolerance is None:
        return 0.0
    if tolerance < 0:
        raise ValueError("the symmetry tolerance must be non-negative or None")
    residuals = little_group_residuals(build, masses, symmetry, grid)
    if not residuals:
        return 0.0
    points = np.asarray(
        [grid.full.points[int(star.representative)] for star in grid.stars], dtype=float
    )
    matrices = build(points)
    scale = float(np.max(np.abs(matrices))) if matrices.size else 0.0
    worst = max(residuals, key=lambda entry: entry.residual)
    allowed = tolerance * scale if scale > 0 else tolerance
    if worst.residual > allowed:
        raise SymmetryViolationError(
            f"{context}: the force constants break the crystal symmetry of their own "
            f"primitive cell: operation {worst.operation} at label {list(worst.label)} leaves "
            f"a residual of {worst.residual:.6e} against an allowed {allowed:.6e} "
            f"(scale {scale:.6e}, relative tolerance {tolerance:.3e}). The dynamical matrix "
            "is not covariant, so the star expansion would mix inequivalent points; fix the "
            "force constants or raise symmetry_tolerance explicitly."
        )
    return float(worst.residual)


def require_hermitian(
    matrix: np.ndarray,
    *,
    scale: float,
    tolerance: float | None,
    label: tuple[int, int, int],
    operation: int,
    context: str = "the model",
) -> None:
    """Raise unless an expanded matrix is Hermitian within a relative tolerance.

    A covariant expansion of a Hermitian ``D(q)`` is Hermitian; a violation means the input
    force constants were not, and the real-space sum built from it would be complex.

    Non-finite data is refused before the comparison and regardless of ``tolerance``: a NaN makes
    ``residual > allowed`` false, so a gate that only compares would admit it silently.
    """
    from mlfcs.reciprocal.modes import require_finite

    values = require_finite(matrix, role="expanded matrix", context=context)
    if not np.isfinite(scale):
        raise SymmetryViolationError(
            f"{context}: the comparison scale is not finite ({scale!r}); non-finite data is "
            "refused whether or not the tolerance comparisons are switched off"
        )
    residual = float(np.max(np.abs(values - values.conj().swapaxes(-1, -2))))
    if not np.isfinite(residual):
        raise SymmetryViolationError(
            f"{context}: the Hermiticity residual for label {list(label)} with operation "
            f"{operation} is not finite ({residual!r})"
        )
    if tolerance is None:
        return
    allowed = tolerance * scale if scale > 0 else tolerance
    if not np.isfinite(allowed):
        raise SymmetryViolationError(
            f"{context}: the allowed Hermiticity residual is not finite ({allowed!r})"
        )
    if residual > allowed:
        raise SymmetryViolationError(
            f"{context}: the matrix expanded for label {list(label)} with operation "
            f"{operation} is not Hermitian: residual {residual:.6e} against an allowed "
            f"{allowed:.6e} (scale {scale:.6e}). The source force constants are not "
            "Hermitian, so the covariance would not be real."
        )


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
    return _residuals_at(
        lambda qpoint: dynamical_matrix(terms, masses, qpoint),
        masses,
        symmetry,
        labels,
        denominator,
        operations,
    )


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
    "StarCovarianceResidual",
    "StarMemberAction",
    "conjugate_matrix",
    "covariance_residuals",
    "displacement_representation",
    "expand_star_matrices",
    "expand_star_values",
    "little_group_residuals",
    "maximum_covariance_residual",
    "require_little_group_covariance",
    "star_member_action",
    "star_member_gauge",
    "star_member_operator",
    "validate_site_masses",
    "validate_symmetry_tolerance",
    "validate_symprec",
]
