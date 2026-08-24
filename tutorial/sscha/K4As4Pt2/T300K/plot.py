"""Plot harmonic and newly generated SSCHA K4As4Pt2 bands together."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure

CASE = Path(__file__).resolve().parent
INPUT = CASE
RESULTS = CASE


def _phonon(force_constants: np.ndarray | Path) -> Phonopy:
    cell, _ = read_crystal_structure(filename=str(INPUT / "supercell.vasp"), interface_mode="vasp")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    if isinstance(force_constants, Path):
        phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(force_constants))
    else:
        phonon.force_constants = force_constants
    return phonon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=CASE / "harmonic-vs-sscha.png")
    args = parser.parse_args()
    harmonic = _phonon(CASE / "FORCE_CONSTANTS_BOOTSTRAP")
    sscha = _phonon(CASE / "FORCE_CONSTANTS_SSCHA")
    structure = (
        np.asarray(harmonic.primitive.cell),
        np.asarray(harmonic.primitive.scaled_positions),
        np.asarray(harmonic.primitive.numbers),
    )
    path_data = seekpath.get_path(structure, recipe="hpkot")
    points = path_data["point_coords"]
    labels = path_data["path"]
    paths = [np.linspace(points[start], points[end], 151) for start, end in labels]
    connections = [True] * len(paths)
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for phonon, color, label in (
        (harmonic, "#3b6ea8", "Harmonic"),
        (sscha, "#c26d5a", "SSCHA 300 K"),
    ):
        phonon.run_band_structure(paths, path_connections=connections)
        labeled = False
        for distance, frequencies in zip(
            phonon.band_structure.distances, phonon.band_structure.frequencies, strict=True
        ):
            for branch_index, branch in enumerate(np.asarray(frequencies).T):
                axis.plot(
                    distance,
                    branch,
                    color=color,
                    linewidth=1.0,
                    label=label if not labeled and branch_index == 0 else None,
                )
            labeled = True
    ticks = {}
    for distance, (start, end) in zip(harmonic.band_structure.distances, labels, strict=True):
        ticks.setdefault(float(distance[0]), start)
        ticks.setdefault(float(distance[-1]), end)
    axis.set_xticks(list(ticks), [r"$\Gamma$" if x == "GAMMA" else x for x in ticks.values()])
    axis.axhline(0.0, color="#777777", linewidth=0.6, linestyle="--")
    axis.set_xlabel("Wave vector path")
    axis.set_ylabel("Frequency (THz)")
    axis.set_title("K4As4Pt2: harmonic and 300 K SSCHA phonon bands")
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.5)
    axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
