#!/usr/bin/env python3
"""Prepare ten 0.01 Å Gaussian SnSe snapshots in a 2x4x4 supercell."""

from pathlib import Path

import numpy as np
from ase.io import read, write
from hiphive import ForceConstantPotential
from hiphive.calculators import ForceConstantCalculator

from mlfcs import build_supercell, perturb_structures

ROOT = Path(__file__).resolve().parent
FCP = ROOT.parent / "input" / "fcp_cm16_rfe-ridge_nf-3000_alpha-1.0.pickle"
SUPERCELL_MATRIX = np.diag((2, 4, 4))


def main() -> None:
    primitive = read(ROOT / "primitive.vasp")
    supercell = build_supercell(primitive, SUPERCELL_MATRIX)
    write(ROOT / "supercell.vasp", supercell, format="vasp", direct=True, sort=False, vasp5=True)
    snapshots = perturb_structures(
        supercell,
        snapshots=10,
        method="gaussian",
        displacement=0.01,
        random_seed=42,
    )
    potential = ForceConstantPotential.read(str(FCP))
    calculator = ForceConstantCalculator(potential.get_force_constants(supercell))
    for snapshot in snapshots:
        snapshot.calc = calculator
        snapshot.new_array("forces", snapshot.get_forces())
        snapshot.calc = None
    write(ROOT / "training.extxyz", snapshots, format="extxyz")
    print(f"wrote {len(snapshots)} snapshots for {len(supercell)} atoms")


if __name__ == "__main__":
    main()

