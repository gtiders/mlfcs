"""Method-independent harmonic-mode utilities shared by SSCHA and SCPH."""

from __future__ import annotations

import numpy as np
from ase import units

HBAR_ASE = units._hbar * units.J * units.s
OMEGA_TO_THZ = units.s / (2 * np.pi * 1e12)


def regular_qpoints(mesh: tuple[int, int, int]):
    """Yield fractional q points on a half-open regular mesh."""
    for i in range(mesh[0]):
        for j in range(mesh[1]):
            for k in range(mesh[2]):
                yield np.array((i / mesh[0], j / mesh[1], k / mesh[2]), dtype=float)


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


__all__ = ["HBAR_ASE", "OMEGA_TO_THZ", "mode_sigma", "regular_qpoints"]
