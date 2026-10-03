#!/usr/bin/env python3
"""Neighbour-shell cutoff advisory for one periodic structure.

Fuses the shell spectrum of the rewritten core (all site pairs over every
periodic translation, aggregated by the one declared ``symprec``) with the
supercell advisory of the retired ``estimate_cutoff`` standalone script: the
reported cutoff between two shells is their midpoint, which captures exactly
the shells below it no matter where floating point lands on a shell radius,
and the Wigner-Seitz inradius reports how far a neighbour list may reach in
this cell before periodic images duplicate.
"""

from __future__ import annotations

import math
import operator

import numpy as np
import spglib
from ase import Atoms
from ase.units import Bohr


class EstimateCutoff:
    """Midpoint cutoff advisory between the neighbour shells of one structure.

    The spectrum counts every site pair over every periodic translation of the
    input cell, so it is a property of the crystal, not of the cell size; the
    Wigner-Seitz inradius is a property of the given cell and bounds where a
    neighbour list on this exact cell stops seeing duplicate images.
    """

    __slots__ = ("_probe", "_shells", "atoms", "symprec")

    def __init__(self, atoms: Atoms, *, symprec: float = 1e-5):
        if not isinstance(atoms, Atoms):
            raise TypeError("atoms must be an ASE Atoms object")
        if not bool(np.all(atoms.pbc)):
            raise ValueError("structure must be periodic in all three directions")
        if not np.isfinite(symprec) or symprec <= 0.0:
            raise ValueError("symprec must be a positive finite length in angstrom")
        cell = np.asarray(atoms.cell, dtype=np.float64)
        if (
            cell.shape != (3, 3)
            or not np.all(np.isfinite(cell))
            or float(np.linalg.det(cell)) == 0.0
        ):
            raise ValueError("structure requires a finite nonsingular cell")
        self.atoms = atoms.copy()  # Detached: later caller mutations are invisible.
        self.symprec = float(symprec)
        self._shells: tuple[float, ...] = ()
        self._probe = 0.0

    def get(self, n: int, *, units: str = "A") -> float:
        """Return the cutoff midpoint between neighbour shells ``n`` and ``n+1``."""
        n = operator.index(n)
        if n < 1:
            raise ValueError("shell index must be at least one")
        if units not in ("A", "Bohr"):
            raise ValueError("units must be 'A' or 'Bohr'")
        shells = self._spectrum(n + 1)
        if len(shells) < n + 1:
            raise ValueError(
                f"the structure resolves only {len(shells)} shells; "
                f"shell {n} has no successor to build a midpoint with"
            )
        return _in_unit(0.5 * (shells[n - 1] + shells[n]), units)

    def report(self, max_shells: int = 10, *, units: str = "A") -> None:
        """Print the shell table for the first ``max_shells`` midpoints."""
        max_shells = operator.index(max_shells)
        if max_shells < 1:
            raise ValueError("max_shells must be at least one")
        if units not in ("A", "Bohr"):
            raise ValueError("units must be 'A' or 'Bohr'")
        shells = self._spectrum(max_shells + 1)
        inradius = _in_unit(_ws_inradius(np.asarray(self.atoms.cell, dtype=np.float64)), units)
        symbols = " ".join(dict.fromkeys(self.atoms.get_chemical_symbols()))
        print(f"Structure: {len(self.atoms)} atoms ({symbols}); symprec {self.symprec:g} A")
        print(f"WS inradius (max cutoff without duplicate images): {inradius:.6f} {units}")
        print()
        print(f"{'Shell':>5s}  {'Cutoff':>12s}  {'d_n':>12s}  {'d_n+1':>12s}   ({units})")
        print("-" * 58)
        rows = 0
        for index in range(min(max_shells, len(shells) - 1)):
            midpoint = _in_unit(0.5 * (shells[index] + shells[index + 1]), units)
            print(
                f"{index + 1:5d}  {midpoint:12.6f}  "
                f"{_in_unit(shells[index], units):12.6f}  "
                f"{_in_unit(shells[index + 1], units):12.6f}"
            )
            rows += 1
        if rows == 0:
            print("(fewer than two shells resolved within the probe budget)")

    def max_cutoff(self, *, units: str = "A") -> float:
        """Return the largest midpoint cutoff this cell supports.

        The two largest shells inside the Wigner-Seitz inradius bracket the
        last safe midpoint: beyond it the next shell or a duplicate periodic
        image is closer than the cutoff itself.
        """
        if units not in ("A", "Bohr"):
            raise ValueError("units must be 'A' or 'Bohr'")
        inradius = _ws_inradius(np.asarray(self.atoms.cell, dtype=np.float64))
        shells = _aggregate(_pair_distances(self.atoms, inradius), self.symprec)
        if len(shells) < 2:
            raise ValueError(
                f"the cell resolves only {len(shells)} shell(s) inside its Wigner-Seitz "
                f"inradius {inradius:.6f} A and cannot separate two shells"
            )
        return _in_unit(0.5 * (shells[-2] + shells[-1]), units)

    def _spectrum(self, need: int) -> tuple[float, ...]:
        if len(self._shells) >= need:
            return self._shells
        probe = max(self._probe, 2.0 * self.symprec)
        for _ in range(64):
            shells = _aggregate(_pair_distances(self.atoms, probe), self.symprec)
            if len(shells) >= need:
                # Shells within a radius are the globally smallest ones, so the
                # prefix is final even though later probes reach farther.
                self._shells, self._probe = shells, probe
                return shells
            probe *= 1.5
        raise RuntimeError(
            f"neighbour-shell enumeration did not reach {need} shells within the probe budget"
        )


def _in_unit(value: float, units: str) -> float:
    """Convert one angstrom length into the requested unit."""
    return value if units == "A" else value / Bohr


def _pair_distances(atoms: Atoms, radius: float) -> np.ndarray:
    """Distances of all site pairs over all translations inside the radius sphere."""
    cell = np.asarray(atoms.cell, dtype=np.float64)
    inverse = np.linalg.inv(cell)
    bounds = [math.ceil(radius * float(np.linalg.norm(inverse[:, j]))) + 1 for j in range(3)]
    shifts = np.stack(
        np.meshgrid(*[np.arange(-b, b + 1) for b in bounds], indexing="ij"), axis=-1
    ).reshape(-1, 3)
    offsets = shifts @ cell
    positions = atoms.get_positions()
    norms = []
    for offset in offsets:
        delta = positions[None, :, :] + offset - positions[:, None, :]
        norms.append(np.linalg.norm(delta, axis=-1).reshape(-1))
    values = np.concatenate(norms)
    return values[(values > 0.0) & (values <= radius)]


def _aggregate(values: np.ndarray, symprec: float) -> tuple[float, ...]:
    """Sorted distances with entries less than ``symprec`` apart merged."""
    shells: list[float] = []
    for value in sorted(float(item) for item in values):
        if not shells or value - shells[-1] >= symprec:
            shells.append(value)
    return tuple(shells)


def _ws_inradius(cell: np.ndarray) -> float:
    """Half the shortest non-zero lattice vector of the Delaunay-reduced cell."""
    reduced = spglib.delaunay_reduce(np.array(cell, dtype=float))
    if reduced is None:
        reduced = np.array(cell, dtype=float)

    sigma_min = float(np.min(np.linalg.svd(reduced, compute_uv=False)))
    best_norm = math.inf
    radius = 1
    while True:
        found = False
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                for k in range(-radius, radius + 1):
                    if max(abs(i), abs(j), abs(k)) != radius:
                        continue
                    coeff = np.array([i, j, k], dtype=int)
                    if not np.any(coeff):
                        continue
                    norm = float(np.linalg.norm(coeff @ reduced))
                    if norm < best_norm - 1e-12:
                        best_norm = norm
                        found = True
        if found and best_norm <= sigma_min * (radius + 1):
            break
        radius += 1

    return best_norm / 2.0


__all__ = ["EstimateCutoff"]
