"""Plot same-fit harmonic and loop-SCPH K4As4Pt2 phonon bands."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from ase.io import read
from matplotlib import colormaps
from matplotlib.lines import Line2D
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure

from mlfcs import read_hdf5, realize_force_constants

ROOT = Path(__file__).resolve().parent
SUPERCELL = ROOT / "input/supercell.vasp"


def _phonon(source: Path) -> Phonopy:
    cell, _ = read_crystal_structure(filename=str(SUPERCELL), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"cannot read {SUPERCELL}")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = realize_force_constants(
        read_hdf5(source), reference=read(SUPERCELL)
    ).materialize(2)
    return phonon


def _band_data(source: Path):
    phonon = _phonon(source)
    primitive = phonon.primitive
    path = seekpath.get_path(
        (primitive.cell, primitive.scaled_positions, primitive.numbers), recipe="hpkot"
    )
    labels = path["path"]
    paths = [
        np.linspace(path["point_coords"][start], path["point_coords"][end], 101)
        for start, end in labels
    ]
    connections = [
        index + 1 < len(paths) and labels[index][1] == labels[index + 1][0]
        for index in range(len(paths))
    ]
    phonon.run_band_structure(paths, path_connections=connections)
    return phonon.band_structure, labels


def _pretty_label(label: str) -> str:
    return {"GAMMA": r"$\Gamma$", "SIGMA_0": r"$\Sigma_0$", "E_0": r"$E_0$"}.get(
        label, label
    )


def main() -> None:
    sources = sorted(
        (int(path.name[1:-1]), path / "mlfcs.h5")
        for path in ROOT.glob("T*K")
        if path.name[1:-1].isdigit() and (path / "mlfcs.h5").is_file()
    )
    if not sources:
        raise FileNotFoundError("run.py must create at least one T*K/mlfcs.h5 result")

    harmonic, labels = _band_data(ROOT / "source/mlfcs.h5")
    figure, axis = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    ticks: dict[float, str] = {}
    for distance, frequencies, (start, end) in zip(
        harmonic.distances, harmonic.frequencies, labels, strict=True
    ):
        ticks.setdefault(float(distance[0]), _pretty_label(start))
        ticks.setdefault(float(distance[-1]), _pretty_label(end))
        axis.plot(
            distance,
            np.asarray(frequencies),
            color="#6b7280",
            linestyle="--",
            linewidth=1.0,
        )

    colors = colormaps["viridis"](np.linspace(0.18, 0.88, len(sources)))
    legend = [Line2D([0], [0], color="#6b7280", linestyle="--", label="Harmonic")]
    for (temperature, source), color in zip(sources, colors, strict=True):
        band, _ = _band_data(source)
        for distance, frequencies in zip(band.distances, band.frequencies, strict=True):
            axis.plot(distance, np.asarray(frequencies), color=color, linewidth=1.05)
        legend.append(Line2D([0], [0], color=color, label=f"{temperature} K"))

    for location in ticks:
        axis.axvline(location, color="#d1d5db", linewidth=0.55, zorder=0)
    axis.axhline(0.0, color="#374151", linewidth=0.7, linestyle=":", zorder=0)
    axis.set_xticks(list(ticks), list(ticks.values()))
    axis.set_xlim(float(harmonic.distances[0][0]), float(harmonic.distances[-1][-1]))
    axis.set_xlabel("High-symmetry path")
    axis.set_ylabel("Frequency (THz)")
    axis.set_title("K4As4Pt2 harmonic and loop-SCPH phonon bands")
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.5)
    axis.legend(handles=legend, frameon=False, ncol=2)
    figure.savefig(ROOT / "harmonic-vs-scph.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
