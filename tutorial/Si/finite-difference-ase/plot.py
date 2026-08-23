"""Plot the Si phonon band from SPOSCAR and phonopy FC2 HDF5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.file_IO import read_force_constants_hdf5
from phonopy.interface.calculator import read_crystal_structure


STRUCTURE = Path("SPOSCAR")
FORCE_CONSTANTS = Path("force_constants.hdf5")
OUTPUT = Path("phonon-band.png")


def main() -> None:
    cell, _ = read_crystal_structure(filename=str(STRUCTURE), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"phonopy could not read {STRUCTURE}")

    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = read_force_constants_hdf5(str(FORCE_CONSTANTS))

    primitive = phonon.primitive
    path_data = seekpath.get_path(
        (
            np.asarray(primitive.cell),
            np.asarray(primitive.scaled_positions),
            np.asarray(primitive.numbers),
        ),
        recipe="hpkot",
    )
    segments = path_data["path"]
    point_coords = path_data["point_coords"]
    paths = [
        np.linspace(point_coords[start], point_coords[end], 101)
        for start, end in segments
    ]
    connections = [
        index + 1 < len(segments) and end == segments[index + 1][0]
        for index, (start, end) in enumerate(segments)
    ]
    phonon.run_band_structure(paths, path_connections=connections)
    band = phonon.band_structure

    ticks: dict[float, str] = {}
    for (start, end), distance in zip(segments, band.distances, strict=True):
        for location, label in ((float(distance[0]), start), (float(distance[-1]), end)):
            if location in ticks and ticks[location] != label:
                ticks[location] += f"|{label}"
            else:
                ticks[location] = label

    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for distance, frequencies in zip(band.distances, band.frequencies, strict=True):
        for branch in np.asarray(frequencies).T:
            axis.plot(distance, branch, color="#176b87", linewidth=0.9)
    for location in ticks:
        axis.axvline(location, color="#cbd5e1", linewidth=0.65, zorder=0)
    axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
    axis.set_xticks(list(ticks), list(ticks.values()))
    axis.set_ylabel("Frequency (THz)")
    axis.set_title("Si phonon band structure from MLFCS FC2")
    axis.set_xlim(float(band.distances[0][0]), float(band.distances[-1][-1]))
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "structure": str(STRUCTURE),
        "force_constants": str(FORCE_CONSTANTS),
        "output": str(OUTPUT),
        "primitive_atoms": len(primitive),
        "path": [list(segment) for segment in segments],
        "q_points": sum(len(path) for path in paths),
    }
    Path("phonon-band.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} and phonon-band.json")


if __name__ == "__main__":
    main()
