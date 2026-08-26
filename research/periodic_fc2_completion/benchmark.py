#!/usr/bin/env python3
"""Reproduce the Si/NaCl exact-R versus periodic-FC2 benchmark."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from ase.io import read
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.structure.atoms import PhonopyAtoms

from mlfcs import ForceConstantFitter
from mlfcs.force_constants.dense import expand_compact_fc2

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]


def _phonopy_atoms(atoms):
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def _fit_case(name, primitive_path, reference_path, dataset_path, reference_fc2, cutoff):
    primitive = read(primitive_path)
    reference = read(reference_path)
    structures = read(dataset_path, index=":")
    values = {}
    arrays = {}
    for label, enabled in (("exact", False), ("hybrid", True)):
        fitter = ForceConstantFitter(
            primitive,
            reference,
            orders=(2,),
            cutoffs={2: cutoff},
            max_body_orders={2: 2},
            periodic_fc2_completion=enabled,
        )
        started = time.perf_counter()
        gram = fitter.prepare_gram(structures, acoustic_sum_rule=True)
        result = fitter.fit(
            gram,
            tolerance=1e-10,
            max_iterations=10_000,
        )
        elapsed = time.perf_counter() - started
        compact = result.force_constants.materialize(2)
        full = expand_compact_fc2(compact, fitter.reference)
        arrays[label] = full
        representatives = np.asarray(
            [fitter.geometry.index.representative(i) for i in range(len(fitter.primitive))]
        )
        reference_compact = reference_fc2[representatives]
        values[label] = {
            "fit_seconds": elapsed,
            "force_rmse_ev_per_angstrom": result.training_force_rmse,
            "relative_force_error": result.training_relative_force_error,
            "exact_parameters": fitter.n_parameters,
            "completion_parameters": (
                result.periodic_fc2_rank.completion_dimension
                if result.periodic_fc2_rank is not None
                else 0
            ),
            "phonopy_fc2_relative": float(
                np.linalg.norm(compact - reference_compact) / np.linalg.norm(reference_compact)
            ),
        }
    arrays["phonopy"] = reference_fc2
    return {"name": name, "reference": reference, "metrics": values, "fc2": arrays}


def _bands(reference, fc2):
    phonon = Phonopy(_phonopy_atoms(reference), np.eye(3, dtype=int), primitive_matrix="auto")
    phonon.force_constants = fc2
    primitive = phonon.primitive
    path = seekpath.get_path(
        (primitive.cell, primitive.scaled_positions, primitive.numbers), recipe="hpkot"
    )
    segments = path["path"]
    qpaths = [
        np.linspace(path["point_coords"][start], path["point_coords"][end], 81)
        for start, end in segments
    ]
    connections = [
        i + 1 < len(segments) and end == segments[i + 1][0]
        for i, (_start, end) in enumerate(segments)
    ]
    phonon.run_band_structure(qpaths, path_connections=connections)
    return path, phonon.band_structure


def _plot(cases):
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.1))
    styles = {
        "phonopy": ("Phonopy direct FC2", "#334e68", 1.5, "-"),
        "exact": ("MLFCS exact-$R$", "#d17a52", 1.0, "--"),
        "hybrid": ("MLFCS exact-$R$ + completion", "#3f8f72", 1.2, "-"),
    }
    for axis, case in zip(axes, cases, strict=True):
        bands = {}
        path = None
        for label in ("phonopy", "exact", "hybrid"):
            path, bands[label] = _bands(case["reference"], case["fc2"][label])
            _title, color, width, style = styles[label]
            for distance, frequencies in zip(
                bands[label].distances, bands[label].frequencies, strict=True
            ):
                axis.plot(distance, frequencies, color=color, linewidth=width, linestyle=style)
        ticks = {}
        for (start, end), distance in zip(path["path"], bands["phonopy"].distances, strict=True):
            for position, label in ((distance[0], start), (distance[-1], end)):
                label = "Γ" if label == "GAMMA" else label
                ticks.setdefault(float(position), label)
        for position in ticks:
            axis.axvline(position, color="#d9e2ec", linewidth=0.55, zorder=0)
        axis.axhline(0, color="#829ab1", linewidth=0.6, linestyle=":")
        axis.set_xticks(list(ticks), list(ticks.values()))
        axis.set_xlim(min(ticks), max(ticks))
        axis.set_title(case["name"])
        axis.set_ylabel("Frequency (THz)")
        axis.grid(axis="y", color="#e9eef2", linewidth=0.45)
        reference_values = np.concatenate(bands["phonopy"].frequencies)
        for label in ("exact", "hybrid"):
            difference = np.concatenate(bands[label].frequencies) - reference_values
            case["metrics"][label]["phonon_rms_difference_thz"] = float(
                np.sqrt(np.mean(difference**2))
            )
    handles = [
        plt.Line2D([], [], color=color, linewidth=width, linestyle=style, label=title)
        for title, color, width, style in styles.values()
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    figure.savefig(ROOT / "phonon-comparison.png", dpi=220)
    plt.close(figure)


def main():
    si = REPOSITORY / "tutorial" / "Si"
    nacl = REPOSITORY / "tutorial" / "long-range-electrostatics" / "NaCl"
    with __import__("h5py").File(si / "finite-difference-ase" / "force_constants.hdf5") as h:
        si_reference = np.asarray(h["force_constants"])
    cases = [
        _fit_case(
            "Si (128 atoms)",
            si / "force-fitting-ase" / "POSCAR.vasp",
            si / "force-fitting-ase" / "SPOSCAR",
            si / "force-fitting-ase" / "train.extxyz",
            si_reference,
            None,
        ),
        _fit_case(
            "NaCl (512 atoms, no NAC)",
            nacl / "primitive.vasp",
            nacl / "supercell.vasp",
            nacl / "training-total.extxyz",
            parse_FORCE_CONSTANTS(nacl / "FORCE_CONSTANTS_PHONOPY"),
            11.0,
        ),
    ]
    _plot(cases)
    output = {case["name"]: case["metrics"] for case in cases}
    (ROOT / "benchmark-results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
