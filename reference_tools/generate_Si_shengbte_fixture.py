"""Maintenance utility to derive the Si FC3 fixture from local VASP calculations."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from ase.io import read


def _parse_thirdorder(path: Path, primitive_cell: np.ndarray):
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    blocks = int(lines[0][0])
    if len(lines) != 1 + 31 * blocks:
        raise ValueError("unexpected FORCE_CONSTANTS_3RD line count")
    inverse = np.linalg.inv(primitive_cell)
    translations = np.empty((blocks, 2, 3), dtype=np.int8)
    atoms = np.empty((blocks, 3), dtype=np.int8)
    values = np.empty((blocks, 3, 3, 3), dtype=float)
    for block in range(blocks):
        chunk = lines[1 + 31 * block : 1 + 31 * (block + 1)]
        if int(chunk[0][0]) != block + 1:
            raise ValueError(f"unexpected block number at index {block}")
        cartesian = np.asarray(chunk[1:3], dtype=float)
        fractional = np.rint(cartesian @ inverse).astype(np.int8)
        if not np.allclose(fractional @ primitive_cell, cartesian, atol=1e-8, rtol=0):
            raise ValueError(f"block {block + 1} translation is not a primitive lattice vector")
        translations[block] = np.mod(fractional, 3)
        atoms[block] = np.asarray(chunk[3], dtype=np.int8)
        for component, line in enumerate(chunk[4:]):
            directions = tuple(int(value) - 1 for value in line[:3])
            values[(block, *directions)] = float(line[3])
    return translations, atoms, values


def _combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generate(mlfcs_root: Path, thirdorder_root: Path, target: Path) -> None:
    """Capture grouped MLFCS forces and the independent thirdorder output."""
    unitcell = read(mlfcs_root / "POSCAR-unitcell")
    force_paths = [
        mlfcs_root / "calculations" / f"{index:03d}" / "vasprun.xml" for index in range(1, 169)
    ]
    forces = np.asarray(
        [read(path, format="vasp-xml", index=-1).get_forces() for path in force_paths]
    )
    reference_path = thirdorder_root / "FORCE_CONSTANTS_3RD"
    translations, atoms, values = _parse_thirdorder(reference_path, np.asarray(unitcell.cell))

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        unitcell_numbers=unitcell.numbers,
        unitcell_cell=np.asarray(unitcell.cell),
        unitcell_scaled_positions=unitcell.get_scaled_positions(),
        mlfcs_forces_grouped=forces,
        mlfcs_plan_hash=np.asarray(
            "5df41821ee9db73b0f008535f9a6c681eb2cb5581c91b6fe8d10c6d9be8e0676"
        ),
        reference_translations_mod_supercell=translations,
        reference_primitive_atoms=atoms,
        reference_fc3=values,
        mlfcs_vasprun_combined_sha256=np.asarray(_combined_sha256(force_paths)),
        thirdorder_file_sha256=np.asarray(hashlib.sha256(reference_path.read_bytes()).hexdigest()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mlfcs_root", type=Path)
    parser.add_argument("thirdorder_root", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    generate(args.mlfcs_root, args.thirdorder_root, args.target)


if __name__ == "__main__":
    main()
