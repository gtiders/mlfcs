"""Method-independent harmonic-mode utilities shared by SSCHA and SCPH."""

from __future__ import annotations

import numpy as np
from ase import units

from mlfcs.core.integer_lattice import (
    IntegerLatticeQuotient,
    adjugate_3x3,
    determinant_3x3,
    normalize_supercell_matrix,
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
    representatives = IntegerLatticeQuotient(matrix.T).representatives
    numerators = representatives @ adjugate_3x3(matrix).T
    points = np.mod(numerators, determinant).astype(float) / determinant
    if len(np.unique(np.round(points, decimals=14), axis=0)) != determinant:
        raise RuntimeError("reciprocal quotient contains duplicate q points")
    if not np.allclose(points @ matrix.T, np.rint(points @ matrix.T), atol=1e-12, rtol=0.0):
        raise RuntimeError("reciprocal quotient contains an incompatible q point")
    return points


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
