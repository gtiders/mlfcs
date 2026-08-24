"""Plot ASR and Born-Huang/Huang MoS2 phonon bands."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure

ROOT = Path(__file__).resolve().parent


def _bands(supercell: Path, force_constants: Path):
    cell, _ = read_crystal_structure(filename=str(supercell), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"cannot read structure: {supercell}")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(force_constants))
    primitive = phonon.primitive
    path = seekpath.get_path((np.asarray(primitive.cell), primitive.scaled_positions, primitive.numbers), recipe="hpkot")
    paths = [np.linspace(path["point_coords"][a], path["point_coords"][b], 101) for a, b in path["path"]]
    links = [i + 1 < len(paths) and path["path"][i][1] == path["path"][i + 1][0] for i in range(len(paths))]
    phonon.run_band_structure(paths, path_connections=links)
    return path, phonon.band_structure


def main() -> None:
    path, asr = _bands(ROOT / "asr/supercell.vasp", ROOT / "asr/FORCE_CONSTANTS_2ND")
    constrained_path = ROOT / "born-huang-huang/FORCE_CONSTANTS_2ND"
    constrained = _bands(ROOT / "asr/supercell.vasp", constrained_path)[1] if constrained_path.is_file() else None
    ticks, labels = [], []
    for (a, b), distance in zip(path["path"], asr.distances, strict=True):
        ticks.extend((float(distance[0]), float(distance[-1])))
        labels.extend((a, b))
    panels = [(asr, "ASR", "#176b87")]
    if constrained is not None:
        panels.append((constrained, "Born-Huang + Huang", "#a34e25"))
    figure, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), sharey=True, squeeze=False, constrained_layout=True)
    axes = axes[0]
    for axis, (band, title, color) in zip(axes, panels, strict=True):
        for distance, values in zip(band.distances, band.frequencies, strict=True):
            axis.plot(distance, np.asarray(values), color=color, linewidth=0.8)
        axis.set_title(title)
        axis.set_xticks(ticks, labels)
        axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
        axis.set_xlabel("Seekpath high-symmetry path")
    axes[0].set_ylabel("Frequency (THz)")
    output = ROOT / "phonon-bands.png"
    figure.savefig(output, dpi=220, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
