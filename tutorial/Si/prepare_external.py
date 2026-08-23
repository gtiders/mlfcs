"""Write the Si finite-difference structures for an external calculator."""

from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write

from mlfcs import ForceConstantCalculation, build_supercell


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
WORK = ROOT / "work" / "external"


def main() -> None:
    primitive = read(SOURCE / "POSCAR.vasp")
    reference = build_supercell(primitive, (4, 4, 4))
    calculation = ForceConstantCalculation(
        primitive,
        order=2,
        reference=reference,
        cutoff=None,
        displacement=0.01,
    )
    structures = calculation.sow()

    structure_dir = WORK / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    for atoms in structures:
        configuration_id = atoms.info["mlfcs_configuration_id"]
        write(
            structure_dir / f"POSCAR-{configuration_id:05d}",
            atoms,
            format="vasp",
            direct=True,
            sort=False,
            vasp5=True,
        )
    (WORK / "manifest.json").write_text(
        json.dumps(
            {
                "order": 2,
                "cutoff": calculation.cutoff,
                "displacement": calculation.config.displacement,
                "configuration_ids": [
                    atoms.info["mlfcs_configuration_id"] for atoms in structures
                ],
                "reference_atoms": len(reference),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(structures)} structures to {structure_dir}")


if __name__ == "__main__":
    main()
