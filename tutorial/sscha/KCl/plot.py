"""Compare KCl Taylor-SSCHA bands for two explicit reference supercells."""

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
BOOTSTRAP_COLOR = "#4f7db8"
SSCHA_COLOR = "#d87a63"
PHONOPY_COLOR = "#4d9678"
LRC_COLOR = "#8d6cab"


def _bands(directory: Path, name: str):
    cell, _ = read_crystal_structure(filename=str(directory / "supercell.vasp"), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"cannot read {directory / 'supercell.vasp'}")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(directory / name))
    primitive = phonon.primitive
    path = seekpath.get_path((primitive.cell, primitive.scaled_positions, primitive.numbers), recipe="hpkot")
    paths = [np.linspace(path["point_coords"][a], path["point_coords"][b], 101) for a, b in path["path"]]
    links = [i + 1 < len(paths) and path["path"][i][1] == path["path"][i + 1][0] for i in range(len(paths))]
    phonon.run_band_structure(paths, path_connections=links)
    return path, phonon.band_structure


def _plot_band_structure(axis, band, *, color: str) -> None:
    """Draw all path segments without creating one legend entry per branch."""
    for distance, frequencies in zip(band.distances, band.frequencies, strict=True):
        axis.plot(distance, frequencies, color=color, linewidth=1.15)


def _path_ticks(path, distances) -> tuple[list[float], list[str]]:
    """Return one label per path boundary, merging coincident boundaries."""
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
    configurations = (("2x2x2", 6.0), ("4x4x4", 12.0))
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), sharey=True)
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.78, wspace=0.04)
    summary = {}
    for axis, (label, cutoff) in zip(axes, configurations, strict=True):
        directory = ROOT / label
        path, bootstrap = _bands(directory, "FORCE_CONSTANTS_BOOTSTRAP")
        final = _bands(directory, "FORCE_CONSTANTS_SSCHA")[1]
        _plot_band_structure(axis, bootstrap, color=BOOTSTRAP_COLOR)
        _plot_band_structure(axis, final, color=SSCHA_COLOR)
        ticks, labels = _path_ticks(path, bootstrap.distances)
        axis.set_title(f"{label}, cutoff {cutoff:g} Å")
        axis.set_xticks(ticks, labels)
        axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
        for tick in ticks[1:-1]:
            axis.axvline(tick, color="#e2e8f0", linewidth=0.55, zorder=0)
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
        axis.set_xlabel("Seekpath high-symmetry path")
        summary[label] = {
            "cutoff_angstrom": cutoff,
            "supercell_atoms": int(sum(map(int, (directory / "supercell.vasp").read_text().splitlines()[6].split()))),
            "history": json.loads((directory / "history.json").read_text(encoding="utf-8")),
        }
    correction = ROOT / "4x4x4-electrostatic-subtracted"
    phonopy_path, phonopy_harmonic = _bands(correction, "FORCE_CONSTANTS_PHONOPY")
    lrc_path, lrc_harmonic = _bands(correction, "FORCE_CONSTANTS_LRC")
    reference_ticks, _ = _path_ticks(phonopy_path, phonopy_harmonic.distances)
    lrc_ticks, _ = _path_ticks(lrc_path, lrc_harmonic.distances)
    panel_ticks = list(axes[1].get_xticks())
    if not np.allclose(reference_ticks, panel_ticks, rtol=0.0, atol=1e-12):
        raise RuntimeError("phonopy harmonic path differs from the SSCHA path")
    if not np.allclose(lrc_ticks, panel_ticks, rtol=0.0, atol=1e-12):
        raise RuntimeError("long-range-corrected path differs from the SSCHA path")
    _plot_band_structure(axes[1], phonopy_harmonic, color=PHONOPY_COLOR)
    _plot_band_structure(axes[1], lrc_harmonic, color=LRC_COLOR)
    summary["4x4x4-electrostatic-subtracted"] = {
        "preparation": json.loads((correction / "preparation.json").read_text(encoding="utf-8")),
        "fit": json.loads((correction / "metrics.json").read_text(encoding="utf-8")),
        "band_nac": False,
    }
    axes[0].set_ylabel("Frequency (THz)")
    legend_handles = [
        Line2D([], [], color=BOOTSTRAP_COLOR, linewidth=1.8, label="SSCHA bootstrap"),
        Line2D([], [], color=SSCHA_COLOR, linewidth=1.8, label="SSCHA 600 K"),
        Line2D([], [], color=PHONOPY_COLOR, linewidth=1.8, label="Phonopy harmonic, no NAC"),
        Line2D([], [], color=LRC_COLOR, linewidth=1.8, label="MLFCS dipole-subtracted fit, no NAC"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    figure.savefig(ROOT / "sscha-supercell-comparison.png", dpi=220, bbox_inches="tight")
    (ROOT / "comparison.json").write_text(json.dumps(summary, default=str, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
