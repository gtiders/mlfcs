"""Method-independent harmonic-mode utilities shared by SSCHA and SCPH."""

from __future__ import annotations

import numpy as np
from ase import units

from mlfcs.core.integer_lattice import (
    adjugate_3x3,
    determinant_3x3,
    normalize_supercell_matrix,
    residue_key,
)

HBAR_ASE = units._hbar * units.J * units.s
OMEGA_TO_THZ = units.s / (2 * np.pi * 1e12)


def quotient_qpoints(integer_matrix: object) -> np.ndarray:
    """Return reciprocal characters of a general integer supercell matrix.

    For row-vector direct-lattice translations ``R S``, the returned fractional
    reciprocal points obey ``S q in Z^3``.  The array contains exactly
    ``abs(det(S))`` points in a deterministic order.
    """
    matrix = normalize_supercell_matrix(integer_matrix)
    determinant = abs(determinant_3x3(matrix))
    # Keep the historical lexicographic representative order.  The order is
    # numerically observable for finite seeded SSCHA samples, even though the
    # q-point set itself is independent of ordering.
    found: dict[tuple[int, int, int], np.ndarray] = {}
    for values in np.ndindex((determinant, determinant, determinant)):
        candidate = np.asarray(values, dtype=np.int32)
        key = residue_key(candidate, matrix.T)
        if key not in found:
            found[key] = candidate
            if len(found) == determinant:
                break
    if len(found) != determinant:  # pragma: no cover - defensive exact-arithmetic guard
        raise RuntimeError("could not enumerate reciprocal quotient points")
    representatives = np.asarray(list(found.values()), dtype=np.int64)
    numerators = representatives @ adjugate_3x3(matrix).T
    return np.mod(numerators, determinant).astype(float) / determinant


def mode_sigma(
    eigenvalues: np.ndarray,
    *,
    temperature: float,
    statistics: str,
    cutoff_frequency_thz: float = 0.0,
) -> np.ndarray:
    """Return displacement amplitudes for mass-weighted harmonic modes."""
    values = np.asarray(eigenvalues, dtype=float)
    omega = np.sqrt(np.abs(values))
    safe = np.where(omega > 0, omega, 1.0)
    included = omega * OMEGA_TO_THZ > cutoff_frequency_thz
    variance = np.zeros_like(omega)
    if statistics == "classical":
        if temperature > 0:
            variance[included] = units.kB * temperature / safe[included] ** 2
    elif statistics == "quantum":
        energy = HBAR_ASE * safe
        if temperature == 0:
            variance[included] = HBAR_ASE / (2 * safe[included])
        else:
            x = energy / (units.kB * temperature)
            occupation = np.zeros_like(x)
            finite = x < 700
            occupation[finite] = 1.0 / np.expm1(x[finite])
            variance[included] = HBAR_ASE * (0.5 + occupation[included]) / safe[included]
    else:
        raise ValueError("statistics must be 'quantum' or 'classical'")
    return np.sqrt(variance)


__all__ = ["HBAR_ASE", "OMEGA_TO_THZ", "mode_sigma", "quotient_qpoints"]
