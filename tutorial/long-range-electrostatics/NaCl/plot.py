#!/usr/bin/env python3
"""Plot the NaCl long-range electrostatic correction comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from matplotlib.lines import Line2D
from phonopy import Phonopy
from phonopy.file_IO import parse_BORN, parse_FORCE_CONSTANTS
from phonopy.structure.atoms import PhonopyAtoms

ROOT = Path(__file__).resolve().parent
SUPERCELL_MATRIX = np.diag((4, 4, 4))
PRIMITIVE_MATRIX = np.array(
    ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
)
CURVES = (
    ("Phonopy finite difference + NAC", "FORCE_CONSTANTS_PHONOPY", "#315f88", 2.0, "-"),
    ("hiPhive total-force fit + NAC", "FORCE_CONSTANTS_HIPHIVE_TOTAL", "#9a6a46", 1.15, ":"),
    ("MLFCS total-force fit + NAC", "FORCE_CONSTANTS_MLFCS_TOTAL", "#c65f5f", 1.35, "--"),
    ("MLFCS short-force fit + restored LR + NAC", "FORCE_CONSTANTS_MLFCS_RESTORED", "#4a8b73", 1.7, "-"),
)


def _phonopy_atoms(atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def _path(phonon: Phonopy):
    primitive = phonon.primitive
    result = seekpath.get_path(
        (primitive.cell, primitive.scaled_positions, primitive.numbers), recipe="hpkot"
    )
    segments = [
        np.linspace(result["point_coords"][start], result["point_coords"][end], 101)
        for start, end in result["path"]
    ]
    connections = [
        i + 1 < len(segments) and result["path"][i][1] == result["path"][i + 1][0]
        for i in range(len(segments))
    ]
    return result, segments, connections


def _bands(unitcell, filename: str, segments, connections):
    phonon = Phonopy(
        _phonopy_atoms(unitcell),
        SUPERCELL_MATRIX,
        primitive_matrix=PRIMITIVE_MATRIX,
    )
    phonon.force_constants = parse_FORCE_CONSTANTS(ROOT / filename)
    phonon.nac_params = parse_BORN(phonon.primitive, filename=ROOT / "input/BORN")
    phonon.run_band_structure(segments, path_connections=connections)
    return phonon.band_structure


def _ticks(path, distances) -> tuple[list[float], list[str]]:
    positions: list[float] = []
    labels: list[str] = []
    for (start, end), distance in zip(path["path"], distances, strict=True):
        for position, label in ((float(distance[0]), start), (float(distance[-1]), end)):
            label = "Γ" if label == "GAMMA" else label
            if positions and np.isclose(position, positions[-1]):
                if labels[-1] != label:
                    labels[-1] = f"{labels[-1]}|{label}"
            else:
                positions.append(position)
                labels.append(label)
    return positions, labels


def _flat_frequencies(band) -> np.ndarray:
    return np.concatenate([np.asarray(values) for values in band.frequencies], axis=0)


def main() -> None:
    from ase.io import read

    unitcell = read(ROOT / "input/NaCl_unitcell.xyz")
    template = Phonopy(
        _phonopy_atoms(unitcell),
        SUPERCELL_MATRIX,
        primitive_matrix=PRIMITIVE_MATRIX,
    )
    path, segments, connections = _path(template)
    bands = {
        label: _bands(unitcell, filename, segments, connections)
        for label, filename, _, _, _ in CURVES
    }
    reference_label = CURVES[0][0]
    reference = _flat_frequencies(bands[reference_label])
    metrics = {}

    hiphive_corrected = _bands(
        unitcell, "FORCE_CONSTANTS_HIPHIVE_RESTORED", segments, connections
    )

    figure, axis = plt.subplots(figsize=(9.6, 6.2))
    for label, _, color, width, style in CURVES:
        band = bands[label]
        for distance, frequencies in zip(band.distances, band.frequencies, strict=True):
            axis.plot(distance, frequencies, color=color, linewidth=width, linestyle=style)
        difference = _flat_frequencies(band) - reference
        metrics[label] = {
            "rms_frequency_difference_thz": float(np.sqrt(np.mean(difference**2))),
            "maximum_frequency_difference_thz": float(np.max(np.abs(difference))),
        }

    ticks, labels = _ticks(path, bands[reference_label].distances)
    for tick in ticks[1:-1]:
        axis.axvline(tick, color="#dce3e8", linewidth=0.6, zorder=0)
    axis.axhline(0.0, color="#667784", linewidth=0.7, linestyle="--")
    axis.set_xlim(ticks[0], ticks[-1])
    axis.set_xticks(ticks, labels)
    axis.set_xlabel("Seekpath high-symmetry path")
    axis.set_ylabel("Frequency (THz)")
    axis.grid(axis="y", color="#e5e9ec", linewidth=0.55)
    figure.legend(
        handles=[
            Line2D([], [], color=color, linewidth=width, linestyle=style, label=label)
            for label, _, color, width, style in CURVES
        ],
        frameon=False,
        loc="upper center",
        ncols=2,
        bbox_to_anchor=(0.5, 0.985),
    )
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.84)
    figure.savefig(ROOT / "long-range-electrostatics-comparison.png", dpi=240)
    plt.close(figure)

    direct_error = metrics["MLFCS total-force fit + NAC"]["rms_frequency_difference_thz"]
    corrected_error = metrics["MLFCS short-force fit + restored LR + NAC"][
        "rms_frequency_difference_thz"
    ]
    metrics["correction_assessment"] = {
        "corrected_is_closer": bool(corrected_error < direct_error),
        "rms_improvement_fraction": float(
            (direct_error - corrected_error) / direct_error if direct_error else 0.0
        ),
    }
    hiphive_corrected_difference = _flat_frequencies(hiphive_corrected) - reference
    metrics["hiPhive short-force fit + restored LR + NAC"] = {
        "rms_frequency_difference_thz": float(
            np.sqrt(np.mean(hiphive_corrected_difference**2))
        ),
        "maximum_frequency_difference_thz": float(
            np.max(np.abs(hiphive_corrected_difference))
        ),
    }
    metrics["mlfcs_vs_hiphive_fc2"] = {}
    for name in ("TOTAL", "SHORT", "RESTORED"):
        mlfcs_fc2 = parse_FORCE_CONSTANTS(ROOT / f"FORCE_CONSTANTS_MLFCS_{name}")
        hiphive_fc2 = parse_FORCE_CONSTANTS(ROOT / f"FORCE_CONSTANTS_HIPHIVE_{name}")
        difference = mlfcs_fc2 - hiphive_fc2
        metrics["mlfcs_vs_hiphive_fc2"][name.lower()] = {
            "maximum_absolute_difference_ev_per_angstrom2": float(
                np.max(np.abs(difference))
            ),
            "relative_frobenius_difference": float(
                np.linalg.norm(difference) / np.linalg.norm(hiphive_fc2)
            ),
        }
    (ROOT / "band-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {ROOT / 'long-range-electrostatics-comparison.png'}")


if __name__ == "__main__":
    main()
