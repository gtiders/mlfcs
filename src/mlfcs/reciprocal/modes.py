"""Modal thermodynamics on the mass-weighted displacement space.

Three things that look like one are kept apart here:

* the geometric precision ``symprec`` (angstrom) belongs to the structure and the star
  decomposition, not to this module;
* the three Gamma translations are not low-frequency *physics*: they are the null space of the
  mass-weighted translation operator, and they leave the modal space structurally, before any
  ``1 / omega^2`` is evaluated;
* everything else is a mode policy the caller states explicitly -- statistics, temperature, cutoff
  and what to do about a non-positive eigenvalue.

The covariance returned here is the basis-independent ``V diag(sigma^2) V^dagger`` of the modes the
policy accepted.  SCPH and the harmonic sampler both consume it, so neither can drift into its own
Gamma convention, and neither may silently turn a soft or imaginary mode into a stable one with
``sqrt(abs(lambda))``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space

from mlfcs.exceptions import SymmetryViolationError
from mlfcs.reciprocal.statistics import OMEGA_TO_THZ, mode_sigma

_TRANSLATIONS = 3


@dataclass(frozen=True, slots=True)
class ModePolicy:
    """The thermodynamic policy of one modal space, stated by the caller."""

    statistics: str
    temperature: float
    frequency_cutoff_thz: float = 0.0
    imaginary_modes: str = "error"

    def __post_init__(self) -> None:
        if self.statistics not in {"quantum", "classical"}:
            raise ValueError("statistics must be 'quantum' or 'classical'")
        if not np.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not np.isfinite(self.frequency_cutoff_thz) or self.frequency_cutoff_thz < 0:
            raise ValueError("frequency_cutoff_thz must be finite and non-negative")
        if self.imaginary_modes not in {"error", "absolute", "exclude"}:
            raise ValueError("imaginary_modes must be 'error', 'absolute' or 'exclude'")


@dataclass(frozen=True, slots=True)
class ModalCovariance:
    """The covariance of the modes one policy accepted, and what it left out."""

    eigenvalues: np.ndarray
    frequencies_thz: np.ndarray
    included: np.ndarray
    translations: np.ndarray
    matrix: np.ndarray
    excluded_weighted_modes: int
    minimum_included_frequency_thz: float
    policy: ModePolicy

    @property
    def n_modes(self) -> int:
        """Number of modes the modal space actually has."""
        return int(self.eigenvalues.size)


def require_finite(array: object, *, role: str, context: str) -> np.ndarray:
    """Return the array, or raise if it holds NaN or infinity.

    This runs before any ``>``, ``max`` or norm comparison: a NaN makes every comparison false, so
    a gate that compares first and checks later silently admits non-finite physics.
    """
    values = np.asarray(array)
    if values.size and not np.all(np.isfinite(values)):
        bad = int(np.argmax(~np.isfinite(values).ravel()))
        raise SymmetryViolationError(
            f"{context}: {role} contains a non-finite entry at flat index {bad} "
            f"(shape {values.shape}); non-finite data is refused whether or not the tolerance "
            "comparisons are switched off"
        )
    return values


def mass_weighted_translations(masses: np.ndarray) -> np.ndarray:
    """Return the orthonormal mass-weighted translation vectors, one column per axis."""
    weight = np.asarray(masses, dtype=float)
    if weight.ndim != 1 or weight.size == 0:
        raise ValueError("masses must be a non-empty vector")
    if not np.all(np.isfinite(weight)) or np.any(weight <= 0):
        raise ValueError("masses must be finite and positive")
    basis = np.zeros((3 * weight.size, _TRANSLATIONS))
    for axis in range(_TRANSLATIONS):
        basis[axis::3, axis] = np.sqrt(weight)
    return basis / np.linalg.norm(basis, axis=0)


def internal_mode_basis(masses: np.ndarray) -> np.ndarray:
    """Return a deterministic orthonormal basis of the internal displacement subspace.

    The basis is the orthogonal complement of the translations, computed once from the masses.  Any
    orthonormal complement gives the same *projector*, and every quantity built here is a spectral
    function of the projected matrix, so the covariance does not depend on the arbitrary choice of
    complement -- while the choice itself never depends on an eigenvector gauge.
    """
    translations = mass_weighted_translations(masses)
    basis = null_space(translations.T)
    if basis.shape != (3 * len(masses), 3 * len(masses) - _TRANSLATIONS):
        raise RuntimeError(
            f"the internal basis has shape {basis.shape}, expected "
            f"({3 * len(masses)}, {3 * len(masses) - _TRANSLATIONS})"
        )
    return np.ascontiguousarray(basis)


def gamma_acoustic_residual(matrix: np.ndarray, masses: np.ndarray) -> float:
    """Return ``||D(Gamma) B||``, the acoustic sum-rule residual in the mass-weighted space."""
    values = require_finite(matrix, role="D(Gamma)", context="gamma_acoustic_residual")
    translations = mass_weighted_translations(masses)
    return float(np.max(np.linalg.norm(values @ translations, axis=0)))


def _mode_weights(eigenvalues: np.ndarray, policy: ModePolicy, *, context: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(weights, included, excluded)`` of one internal or full mode set."""
    values = require_finite(eigenvalues, role="eigenvalues", context=context)
    frequencies = np.sqrt(np.abs(values)) * np.sign(values) * OMEGA_TO_THZ
    negative = values < 0
    if np.any(negative):
        if policy.imaginary_modes == "error":
            worst = int(np.argmin(values))
            raise SymmetryViolationError(
                f"{context}: non-positive internal mode {worst} with eigenvalue "
                f"{float(values[worst]):.6e} and frequency {float(frequencies[worst]):.6e} THz at "
                f"temperature {policy.temperature} K. A soft or imaginary mode is a physical "
                "statement, not something to hide with abs(); choose imaginary_modes='absolute' or "
                "'exclude' explicitly."
            )
        if policy.imaginary_modes == "exclude":
            values = np.where(negative, 0.0, values)
    sigma = mode_sigma(
        values,
        temperature=policy.temperature,
        statistics=policy.statistics,
        cutoff_frequency_thz=policy.frequency_cutoff_thz,
    )
    require_finite(sigma, role="mode amplitudes", context=context)
    included = sigma > 0.0
    return sigma**2, included, ~included


def modal_eigenpairs(
    matrix: np.ndarray,
    masses: np.ndarray,
    *,
    is_gamma: bool,
    policy: ModePolicy,
    keep_translations: bool = False,
    context: str = "modal_eigenpairs",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(eigenvalues, eigenvectors, frequencies_thz, included)`` of one modal space.

    At Gamma the eigenproblem is solved in the internal subspace and the eigenvectors are lifted
    back into the mass-weighted space, so the three translations are simply *absent* rather than
    present with a huge weight.  The columns of the returned basis are a valid basis of each
    degenerate subspace but not a physical identity, so a caller must compare projectors or
    covariances across a degeneracy, never columns.
    """
    values = require_finite(matrix, role="dynamical matrix", context=context)
    if values.shape[0] != values.shape[1] or values.shape[0] != 3 * len(masses):
        raise ValueError(
            f"{context}: expected a ({3 * len(masses)}, {3 * len(masses)}) matrix, got "
            f"{values.shape}"
        )
    if is_gamma:
        basis = internal_mode_basis(masses)
        eigenvalues, internal = np.linalg.eigh(basis.conj().T @ values @ basis)
        eigenvectors = basis @ internal
        if keep_translations:
            # Re-attach the translations as exact zero modes so a caller that stores one
            # rectangular block per star keeps a uniform width.  They carry eigenvalue zero and
            # a zero amplitude, so they are never sampled, never weighted and never reported as
            # a frequency -- the exclusion stays structural, not a numerical accident.
            translations = mass_weighted_translations(masses)
            eigenvalues = np.concatenate([eigenvalues, np.zeros(_TRANSLATIONS)])
            eigenvectors = np.concatenate([eigenvectors, translations], axis=1)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(values)
    _, included, _ = _mode_weights(eigenvalues, policy, context=context)
    frequencies = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues) * OMEGA_TO_THZ
    return eigenvalues, eigenvectors, frequencies, included


def modal_covariance(
    matrix: np.ndarray,
    masses: np.ndarray,
    *,
    is_gamma: bool,
    policy: ModePolicy,
    context: str = "modal_covariance",
) -> ModalCovariance:
    """Return the covariance of the modes one policy accepts, translations excluded structurally.

    At Gamma the matrix is first projected onto the internal subspace, so the three translations
    never enter ``1 / omega^2`` and never contribute rounding noise to the covariance; elsewhere the
    full matrix is used.  The returned matrix is ``V diag(sigma^2) V^dagger``, finite by
    construction.
    """
    values = require_finite(matrix, role="dynamical matrix", context=context)
    if values.shape[0] != values.shape[1] or values.shape[0] != 3 * len(masses):
        raise ValueError(
            f"{context}: expected a ({3 * len(masses)}, {3 * len(masses)}) matrix, got "
            f"{values.shape}"
        )
    eigenvalues, eigenvectors, frequencies, included = modal_eigenpairs(
        values, masses, is_gamma=is_gamma, policy=policy, context=context
    )
    sigma2, _, excluded = _mode_weights(eigenvalues, policy, context=context)
    covariance = (eigenvectors * sigma2[None, :]) @ eigenvectors.conj().T
    return ModalCovariance(
        eigenvalues=eigenvalues,
        frequencies_thz=frequencies,
        included=included,
        translations=np.zeros(eigenvalues.size, dtype=bool),
        matrix=covariance,
        excluded_weighted_modes=int(np.count_nonzero(excluded)),
        minimum_included_frequency_thz=(
            float(np.min(frequencies[included])) if np.any(included) else float("nan")
        ),
        policy=policy,
    )


def require_star_covariance_matrix(
    build,
    masses: np.ndarray,
    symmetry,
    grid,
    primitive_positions: np.ndarray,
    *,
    tolerance: float | None,
    policy: ModePolicy,
    gamma_star: int | None = None,
    context: str = "covariance",
) -> float:
    """Compare the expanded covariance of every star member with the directly built one.

    A covariant dynamical matrix does not imply a covariant covariance: the classical and quantum
    weights go like ``1 / lambda``, so the error the matrix gate allows is amplified near a small
    eigenvalue.  This certificate builds ``W`` at the representatives, expands it with the same
    stored-label gauge the production path uses, builds ``W`` directly at every full-grid point and
    compares them member by member.
    """
    from mlfcs.reciprocal.symmetry import expand_star_matrices

    representatives = np.asarray(grid.representatives, dtype=np.int64)
    points = grid.full.points
    if gamma_star is None:
        gamma_star = int(grid.full_to_irreducible[int(np.flatnonzero(np.all(grid.full.labels == 0, axis=1))[0])])

    def covariance_at(qpoints: np.ndarray) -> np.ndarray:
        matrices = build(qpoints)
        require_finite(matrices, role="dynamical matrices", context=context)
        out = np.empty_like(matrices)
        for index, qpoint in enumerate(np.asarray(qpoints, dtype=float)):
            is_gamma = bool(np.allclose(qpoint, 0.0, rtol=0.0, atol=1e-12))
            out[index] = modal_covariance(
                matrices[index], masses, is_gamma=is_gamma, policy=policy, context=context
            ).matrix
        return out

    direct = covariance_at(points)
    expanded = expand_star_matrices(
        covariance_at(points[representatives]), grid, symmetry, primitive_positions
    )
    if tolerance is None:
        return 0.0
    require_finite(direct, role="direct covariance", context=context)
    require_finite(expanded, role="expanded covariance", context=context)
    scale = float(
        max(
            np.linalg.norm(direct, ord=np.inf, axis=(-2, -1)).max(),
            np.linalg.norm(expanded, ord=np.inf, axis=(-2, -1)).max(),
        )
    )
    residuals = np.max(np.abs(expanded - direct), axis=(-2, -1))
    worst = int(np.argmax(residuals))
    allowed = tolerance * scale if scale > 0 else tolerance
    if float(residuals[worst]) > allowed:
        star = int(grid.full_to_irreducible[worst])
        representative = int(grid.representatives[star])
        raise SymmetryViolationError(
            f"{context}: the covariance is not covariant on the full star of representative "
            f"{grid.full.labels[representative].tolist()}: member {grid.full.labels[worst].tolist()} "
            f"reached by operation {int(grid.full_operations[worst])} leaves a residual of "
            f"{float(residuals[worst]):.6e} against an allowed {allowed:.6e} (scale {scale:.6e}). "
            "The dynamical matrix may be covariant while 1/lambda amplifies its residual; fix the "
            "force constants or raise symmetry_tolerance explicitly."
        )
    return float(residuals[worst])


__all__ = [
    "ModalCovariance",
    "ModePolicy",
    "gamma_acoustic_residual",
    "internal_mode_basis",
    "mass_weighted_translations",
    "modal_covariance",
    "modal_eigenpairs",
    "require_finite",
    "require_star_covariance_matrix",
]
