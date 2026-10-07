"""Reproducible Gaussian perturbations of periodic atomic positions.

The tool changes geometry only and never evaluates forces. Each instance owns
one random stream, so a fixed seed reproduces its complete sequence of samples.
"""

from __future__ import annotations

import operator

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

# Maximum resampling attempts when minimum-distance rejection is enabled.
_ATTEMPTS = 32


class GaussianPerturbation:
    """Generate reproducible Gaussian perturbations of a periodic structure.

    Cartesian displacements are sampled independently from a zero-mean normal
    distribution with standard deviation ``stdev``. Selected atoms may remain
    fixed, and candidates may be rejected when interatomic distances fall below
    a prescribed threshold.

    The reference geometry is copied at construction and never modified by the
    generator. Each instance owns one random stream, so a fixed ``seed``
    reproduces the complete sequence of generated structures. The tool does not
    evaluate forces.
    """

    __slots__ = ("_atoms", "_rng")

    def __init__(self, atoms: Atoms, *, seed: int | None = None):
        """Create a perturbation generator for a fully periodic reference."""
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object")
        if not bool(np.all(atoms.pbc)):
            raise ValueError("structure must be periodic in all three directions")
        cell = np.asarray(atoms.cell, dtype=np.float64)
        if (
            cell.shape != (3, 3)
            or not np.all(np.isfinite(cell))
            or float(np.linalg.det(cell)) == 0.0
        ):
            raise ValueError("structure requires a finite nonsingular cell")
        self._atoms = atoms.copy()
        self._rng = np.random.default_rng(seed)

    @property
    def atoms(self) -> Atoms:
        """Return a copy of the reference structure."""
        return self._atoms.copy()

    def rattle(self, stdev: float = 0.01, *, indices=None, min_distance="covalent") -> Atoms:
        """Return one Gaussian-perturbed copy of the reference structure.

        Displacements are independent Cartesian normal samples with standard
        deviation ``stdev`` in angstrom. ``indices`` selects atoms that remain
        fixed. ``min_distance`` may be a positive absolute distance,
        ``"covalent"`` for pair thresholds equal to half the sum of the two ASE
        covalent radii, or ``None`` to disable rejection. Rejected candidates
        are resampled from the same random stream.
        """
        stdev = _validate_stdev(stdev)
        mask = _fixed_mask(len(self._atoms), indices)
        kind, value = _validate_min_distance(min_distance)
        violation = None
        for _ in range(_ATTEMPTS):
            displaced = self._atoms.copy()
            noise = self._rng.normal(scale=stdev, size=(len(displaced), 3))
            if mask is not None:
                noise[mask] = 0.0
            displaced.positions += noise
            violation = _closest_violation(displaced, kind, value)
            if violation is None:
                return displaced
        distance, threshold, i, j = violation
        symbols = self._atoms.get_chemical_symbols()
        raise ValueError(
            f"no perturbation kept every pair at or above the minimum distance in "
            f"{_ATTEMPTS} attempts; closest pair {symbols[i]}-{symbols[j]} at "
            f"{distance:.6f} A (threshold {threshold:.6f} A)"
        )

    def ensemble(self, count: int, stdev: float = 0.01, *, indices=None, min_distance="covalent"):
        """Return ``count`` sequential perturbations from the owned random stream."""
        count = operator.index(count)
        if count < 1:
            raise ValueError("count must be at least one")
        return [
            self.rattle(stdev, indices=indices, min_distance=min_distance) for _ in range(count)
        ]


def _validate_stdev(stdev: float) -> float:
    """Return a positive finite Cartesian noise standard deviation in angstrom."""
    value = float(stdev)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("stdev must be a positive finite length in angstrom")
    return value


def _fixed_mask(count: int, indices) -> np.ndarray | None:
    """Return the boolean mask of fixed atoms, validating all indices."""
    if indices is None:
        return None
    mask = np.zeros(count, dtype=bool)
    for index in indices:
        index = operator.index(index)
        if not 0 <= index < count:
            raise IndexError(f"fixed index {index} is outside the structure")
        mask[index] = True
    return mask


def _validate_min_distance(min_distance):
    """Normalize the minimum-distance specification."""
    if min_distance is None:
        return "none", 0.0
    if isinstance(min_distance, str):
        if min_distance != "covalent":
            raise ValueError(
                'min_distance must be "covalent", a positive length in angstrom, or None'
            )
        return "covalent", 0.0
    value = float(min_distance)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("min_distance must be a positive finite length in angstrom")
    return "fixed", value


def _closest_violation(atoms: Atoms, kind: str, value: float):
    """Return the most severe minimum-distance violation, or ``None``."""
    if kind == "none":
        return None
    if kind == "fixed":
        i, j, d = neighbor_list("ijd", atoms, value)
        if len(d) == 0:
            return None
        k = int(np.argmin(d))
        return float(d[k]), float(value), int(i[k]), int(j[k])
    radii = covalent_radii[atoms.numbers]
    cutoff = float(radii.max())
    if cutoff <= 0.0:
        return None
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    if len(d) == 0:
        return None
    threshold = 0.5 * (radii[i] + radii[j])
    bad = d < threshold
    if not np.any(bad):
        return None
    k = int(np.argmin(d[bad] - threshold[bad]))
    return float(d[bad][k]), float(threshold[bad][k]), int(i[bad][k]), int(j[bad][k])


__all__ = ["GaussianPerturbation"]
