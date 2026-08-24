#!/usr/bin/env python3
"""Overlay the two independently fitted harmonic SnSe phonon spectra."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from matplotlib.lines import Line2D
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure

ROOT = Path(__file__).resolve().parent
CASES = (
    ("harmonic-2x4x4", "2x4x4", "#4878a8"),
    ("harmonic-3x5x5", "3x5x5", "#d47c5d"),
)


def _bands(directory: Path):
    cell, _ = read_crystal_structure(str(directory / "supercell.vasp"), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"cannot read {directory / 'supercell.vasp'}")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = parse_FORCE_CONSTANTS(str(directory / "FORCE_CONSTANTS_2ND"))
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
    return path, phonon.band_structure


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


def main() -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    baseline_distances = None
    ticks: list[float] = []
    labels: list[str] = []
    summary = {}
    handles = []
    for directory_name, label, color in CASES:
        directory = ROOT / directory_name
        path, band = _bands(directory)
        flattened = np.concatenate(band.distances)
        if baseline_distances is None:
            baseline_distances = flattened
            ticks, labels = _ticks(path, band.distances)
        elif not np.allclose(flattened, baseline_distances, rtol=0.0, atol=1e-12):
            raise RuntimeError("the two supercells produced different seekpath distances")
        for distance, frequencies in zip(band.distances, band.frequencies, strict=True):
            axis.plot(distance, frequencies, color=color, linewidth=1.05, alpha=0.92)
        handles.append(Line2D([], [], color=color, linewidth=1.8, label=label))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        summary[label] = metrics
    axis.set_xticks(ticks, labels)
    axis.set_ylabel("Frequency (THz)")
    axis.set_title("SnSe harmonic FC2: reference-supercell comparison")
    axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
    for tick in ticks[1:-1]:
        axis.axvline(tick, color="#e2e8f0", linewidth=0.55, zorder=0)
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
    axis.legend(handles=handles, loc="best", frameon=False)
    figure.savefig(ROOT / "harmonic-supercell-comparison.png", dpi=220, bbox_inches="tight")
    (ROOT / "harmonic-comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
