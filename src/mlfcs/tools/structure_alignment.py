"""Explicit alignment of independently produced structures.

This is an *external import* policy, not a structure-identity rule: it may reorder an MD frame or
the output of another program so it matches a reference, and the caller has to name the tolerance
it will accept.  Nothing in the core calculation path calls it -- fitting and finite difference
never silently reorder a training frame -- and the core relation does not share its tolerance.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.optimize import linear_sum_assignment

from mlfcs.structure.periodic_geometry import PeriodicGeometry


def align_structures(
    reference: Atoms,
    atoms: Atoms,
    *,
    tolerance: float,
) -> tuple[Atoms, float]:
    """Explicitly reorder ``atoms`` to ``reference`` and report the residual.

    This utility is intentionally separate from fitting and finite-difference
    APIs. It can be useful for independently produced snapshots, but never
    silently changes the labels supplied to a calculation.
    """
    if len(atoms) != len(reference):
        raise ValueError("structure atom count differs from reference")
    if not np.allclose(atoms.cell, reference.cell, atol=tolerance, rtol=0.0):
        raise ValueError("structure cell differs from reference")
    permutation = np.empty(len(reference), dtype=np.int32)
    maximum = 0.0
    geometry = PeriodicGeometry(reference.cell, reference.pbc)
    for number in np.unique(reference.numbers):
        target = np.flatnonzero(reference.numbers == number)
        source = np.flatnonzero(atoms.numbers == number)
        if len(target) != len(source):
            raise ValueError("structure chemical composition differs from reference")
        delta = atoms.positions[source][None, :, :] - reference.positions[target][:, None, :]
        _, lengths = geometry.mic(delta.reshape(-1, 3))
        cost = lengths.reshape(len(target), len(source))
        rows, columns = linear_sum_assignment(cost)
        maximum = max(maximum, float(np.max(cost[rows, columns], initial=0.0)))
        permutation[target[rows]] = source[columns]
    if maximum > tolerance:
        raise ValueError(
            f"structure cannot be aligned to reference within tolerance; maximum residual {maximum:.3e} Å"
        )
    aligned = atoms[permutation]
    aligned.info.update(atoms.info)
    return aligned, maximum

__all__ = ["align_structures"]
