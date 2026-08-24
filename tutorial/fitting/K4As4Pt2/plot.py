"""Plot the K4As4Pt2 Taylor-fit harmonic bands along a seekpath path."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure

ROOT = Path(__file__).resolve().parent


def main() -> None:
    cell, _ = read_crystal_structure(filename=str(ROOT / "input/supercell.vasp"), interface_mode="vasp")
    if cell is None:
        raise ValueError("cannot read K4As4Pt2 supercell")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(ROOT / "FORCE_CONSTANTS_2ND"))
    primitive = phonon.primitive
    path = seekpath.get_path((primitive.cell, primitive.scaled_positions, primitive.numbers), recipe="hpkot")
    paths = [np.linspace(path["point_coords"][a], path["point_coords"][b], 101) for a, b in path["path"]]
    links = [index + 1 < len(paths) and path["path"][index][1] == path["path"][index + 1][0] for index in range(len(paths))]
    phonon.run_band_structure(paths, path_connections=links)
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for distance, frequencies in zip(phonon.band_structure.distances, phonon.band_structure.frequencies, strict=True):
        axis.plot(distance, frequencies, color="#a34e25", linewidth=0.8)
    axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
    axis.set_xlabel("Seekpath high-symmetry path")
    axis.set_ylabel("Frequency (THz)")
    figure.savefig(ROOT / "phonon-band.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
