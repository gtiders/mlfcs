"""Validate the fitted Si Taylor IFC through the public ASE Calculator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io import read

from mlfcs import MLFCSCalculator

REFERENCE = Path("SPOSCAR")
FORCE_CONSTANTS = Path("fc2-fit-mlfcs.h5")


def main() -> None:
    reference = read(REFERENCE)
    atoms = reference.copy()
    atoms.positions[0] += [0.008, -0.004, 0.006]
    calculator = MLFCSCalculator.from_hdf5(FORCE_CONSTANTS, reference=reference)
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    step = 1e-6
    plus = atoms.copy()
    minus = atoms.copy()
    plus.positions[0, 0] += step
    minus.positions[0, 0] -= step
    plus.calc = MLFCSCalculator.from_hdf5(FORCE_CONSTANTS, reference=reference)
    minus.calc = MLFCSCalculator.from_hdf5(FORCE_CONSTANTS, reference=reference)
    numerical_force = -(plus.get_potential_energy() - minus.get_potential_energy()) / (2 * step)
    difference = abs(numerical_force - forces[0, 0])

    print(f"relative energy: {energy:.12e} eV")
    print(f"maximum force: {np.max(np.abs(forces)):.12e} eV/A")
    print(f"analytic/numerical force difference: {difference:.12e} eV/A")
    if difference > 1e-7:
        raise RuntimeError("ASE Calculator energy-gradient validation failed")


if __name__ == "__main__":
    main()
