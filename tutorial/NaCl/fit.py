"""Compare direct FC2 fitting with explicit dipole-Ewald force subtraction."""

import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from phonopy import Phonopy
from phonopy.file_IO import parse_BORN
from phonopy.structure.atoms import PhonopyAtoms

from mlfcs import ClusterMap, ClusterSpace, FitSystem, Supercell
from mlfcs.core.log_error import configure
from mlfcs.force_constants import write_phonopy
from mlfcs.tools import Ewald

ROOT = Path(__file__).resolve().parent


def fit():
    unit = read(ROOT / "NaCl_unitcell.xyz")
    phonon = Phonopy(
        PhonopyAtoms(
            symbols=unit.get_chemical_symbols(),
            scaled_positions=unit.get_scaled_positions(),
            cell=unit.cell,
        ),
        np.diag([4, 4, 4]),
        primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    primitive = Atoms(
        symbols=phonon.primitive.symbols,
        scaled_positions=phonon.primitive.scaled_positions,
        cell=phonon.primitive.cell,
        pbc=True,
    )
    supercell = Atoms(
        symbols=phonon.supercell.symbols,
        scaled_positions=phonon.supercell.scaled_positions,
        cell=phonon.supercell.cell,
        pbc=True,
    )
    space = ClusterSpace(primitive, cutoffs={2: 11.0}, max_body_orders={2: 2})
    mapping = ClusterMap.build(space, Supercell.from_atoms(space.primitive, supercell))
    frames = read(ROOT / "supercells_with_forces.xyz", index=":")

    direct = FitSystem.from_atoms(mapping, frames)
    direct_model = direct.force_constants(direct.solve(max_steps=10_000))
    direct_model = direct_model.enforce_asr().force_constants
    direct_model.save(ROOT / "fc-direct.mlfcs")
    write_phonopy(
        ROOT / "fc2-direct.hdf5", direct_model.get(2, mapping), mapping, format="phonopy_hdf5"
    )
    print(
        f"Direct fit: relative force error {direct.relative_error(direct_model.parameters()):.6%}"
    )

    nac = parse_BORN(phonon.primitive, filename=str(ROOT / "BORN"))
    ewald = Ewald(space, mapping, nac["born"], nac["dielectric"])
    short_frames = []
    for frame in frames:
        short = frame.copy()
        short.calc = SinglePointCalculator(
            short, forces=frame.get_forces() - ewald.get_forces(frame)
        )
        short_frames.append(short)
    short = FitSystem.from_atoms(mapping, short_frames)
    short_model = short.force_constants(short.solve(max_steps=10_000))
    short_model = short_model.enforce_asr().force_constants
    short_model.save(ROOT / "fc-short.mlfcs")
    write_phonopy(
        ROOT / "fc2-corrected.hdf5",
        short_model.get(2, mapping) + ewald.fc2,
        mapping,
        format="phonopy_hdf5",
    )
    print(
        f"Short-range fit: relative force error {short.relative_error(short_model.parameters()):.6%}"
    )
    print("Saved direct and long-range-corrected FC2 for the phonon comparison.")


if __name__ == "__main__":
    with (ROOT / "fit.log").open("w", encoding="utf-8") as log:
        configure(stream=log)
        try:
            with redirect_stdout(log), redirect_stderr(log):
                fit()
        except BaseException:
            traceback.print_exc(file=log)
            raise
