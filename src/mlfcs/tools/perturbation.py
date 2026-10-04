"""Reproducible Gaussian position perturbations of one periodic structure.

Pure geometry: the tool moves positions and never evaluates forces. One
instance owns a detached copy of the structure and one random stream, so a
fixed ``seed`` reproduces the whole sequence of ``rattle`` and ``ensemble``
calls.
"""

from __future__ import annotations

import operator

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

_ATTEMPTS = 32


class GaussianPerturbation:
    """Reproducible Gaussian Cartesian perturbations of one periodic structure.

    Parameters
    ----------
    atoms : ase.Atoms
        Fully periodic reference with a finite nonsingular cell.
    seed : int, optional
        Seed for one owned NumPy random stream; omission uses fresh entropy.

    Notes
    -----
    Owns a detached structure copy. rattle/ensemble advance the stream, return
    fresh structures and never evaluate forces. Distance rejection resamples
    from the same stream. Fixed atoms are selected by indices in later calls.
    Invalid reference type/geometry raises TypeError or ValueError.
    """

    __slots__ = ("_rng", "atoms")

    def __init__(self, atoms: Atoms, *, seed: int | None = None):
        """Validate periodic geometry, capture a detached reference and initialize the random
        stream.
        """
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
        self.atoms = atoms.copy()  # Detached: later caller mutations are invisible.
        self._rng = np.random.default_rng(seed)

    def rattle(self, stdev: float = 0.01, *, indices=None, min_distance="covalent") -> Atoms:
        """Return one perturbed copy; the input structure is never modified.

        ``indices`` pins those atoms exactly; ``min_distance`` is ``"covalent"``
        (per-pair floor of half the covalent-bond length), a positive absolute
        length in angstrom, or ``None`` to disable the check.
        """
        stdev = _validate_stdev(stdev)
        mask = _fixed_mask(len(self.atoms), indices)
        kind, value = _validate_min_distance(min_distance)
        violation = None
        for _ in range(_ATTEMPTS):
            displaced = self.atoms.copy()
            noise = self._rng.normal(scale=stdev, size=(len(displaced), 3))
            if mask is not None:
                noise[mask] = 0.0
            displaced.positions += noise
            violation = _closest_violation(displaced, kind, value)
            if violation is None:
                return displaced
        distance, threshold, i, j = violation
        symbols = self.atoms.get_chemical_symbols()
        raise ValueError(
            f"no perturbation kept every pair at or above the minimum distance in "
            f"{_ATTEMPTS} attempts; closest pair {symbols[i]}-{symbols[j]} at "
            f"{distance:.6f} A (threshold {threshold:.6f} A)"
        )

    def ensemble(self, count: int, stdev: float = 0.01, *, indices=None, min_distance="covalent"):
        """Sample ``count`` frames from one random stream, in order."""
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
    """Return a mask of pinned atom indices, or None; reject indices outside [0, count)."""
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
    """Return (kind, value) with kind in {"none", "fixed", "covalent"}."""
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
    """Return (distance, threshold, i, j) of the worst violating pair, or None."""
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
