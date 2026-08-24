#!/usr/bin/env python3
"""Plot the effective FC2 from the 300 K joint SnSe fit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure

ROOT = Path(__file__).resolve().parent


def main() -> None:
    cell, _ = read_crystal_structure(str(ROOT / "supercell.vasp"), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"cannot read {ROOT / 'supercell.vasp'}")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = parse_FORCE_CONSTANTS(str(ROOT / "FORCE_CONSTANTS_2ND"))
    primitive = phonon.primitive
    path = seekpath.get_path(
        (primitive.cell, primitive.scaled_positions, primitive.numbers), recipe="hpkot"
    )
    paths = [
        np.linspace(path["point_coords"][start], path["point_coords"][end], 101)
        for start, end in path["path"]
    ]
    connections = [
        index + 1 < len(paths) and path["path"][index][1] == path["path"][index + 1][0]
        for index in range(len(paths))
    ]
    phonon.run_band_structure(paths, path_connections=connections)
    band = phonon.band_structure
    ticks, labels = _ticks(path, band.distances)
    figure, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for distance, frequencies in zip(band.distances, band.frequencies, strict=True):
        axis.plot(distance, frequencies, color="#427aa1", linewidth=1.0)
    axis.set_xticks(ticks, labels)
    axis.set_ylabel("Frequency (THz)")
    axis.set_title("SnSe 300 K effective FC2 from joint FC2+FC3+FC4 fit")
    axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
    figure.savefig(ROOT / "phonon-band.png", dpi=220, bbox_inches="tight")


def _ticks(path, distances) -> tuple[list[float], list[str]]:
    ticks: list[float] = []
    labels: list[str] = []
    for (start, end), distance in zip(path["path"], distances, strict=True):
        for position, label in ((float(distance[0]), start), (float(distance[-1]), end)):
            label = "Γ" if label == "GAMMA" else label
            if ticks and np.isclose(position, ticks[-1]):
                if label != labels[-1]:
                    labels[-1] = f"{labels[-1]}|{label}"
            else:
                ticks.append(position)
                labels.append(label)
    return ticks, labels


if __name__ == "__main__":
    main()

