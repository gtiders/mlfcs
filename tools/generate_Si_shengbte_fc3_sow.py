"""Generate the external VASP sow set for the Si ShengBTE FC3 reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ase.io import read, write

from mlfcs import ForceConstantCalculation

SUPERCELL = (3, 3, 3)
ORDER = 3
NEIGHBOR_SHELL = 6
DISPLACEMENT = 0.01


def generate(source: Path, target: Path) -> None:
    """Write POSCAR files in the exact positional reap order."""
    target.mkdir(parents=True, exist_ok=True)
    existing = list(target.glob("POSCAR-[0-9][0-9][0-9]"))
    if existing:
        raise FileExistsError(f"target already contains {len(existing)} POSCAR files: {target}")

    primitive = read(source, format="vasp")
    calculation = ForceConstantCalculation(
        primitive,
        order=ORDER,
        supercell=SUPERCELL,
        cutoff=-NEIGHBOR_SHELL,
        displacement=DISPLACEMENT,
        jax_platform="cpu",
    )
    structures = calculation.sow(atom_order="grouped")
    write(target / "POSCAR-unitcell", primitive, format="vasp", direct=True, vasp5=True)
    write(
        target / "SPOSCAR",
        calculation.index.group_atoms(calculation.supercell),
        format="vasp",
        direct=True,
        vasp5=True,
    )

    records = []
    for configuration_id, atoms in enumerate(structures, start=1):
        name = f"POSCAR-{configuration_id:03d}"
        path = target / name
        write(path, atoms, format="vasp", direct=True, vasp5=True)
        records.append(
            {
                "configuration_id": configuration_id - 1,
                "filename": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    manifest = {
        "order": ORDER,
        "supercell": list(SUPERCELL),
        "cutoff": -NEIGHBOR_SHELL,
        "cutoff_angstrom": calculation.cutoff,
        "displacement_angstrom": DISPLACEMENT,
        "atom_order": "grouped",
        "plan_hash": calculation.plan.hash,
        "configurations": records,
    }
    (target / "sow-plan.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    generate(args.source, args.target)


if __name__ == "__main__":
    main()
