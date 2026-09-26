"""Compare MLFCS FC2 with the exported Phonopy FC2 on a seekpath band path."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seekpath
from ase.io import read
from matplotlib.lines import Line2D
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS
from phonopy.structure.atoms import PhonopyAtoms
from scipy.constants import angstrom, atomic_mass, electron_volt

from mlfcs.force_constants import ForceConstants
from mlfcs.reciprocal import Harmonic, QGrid, QStars

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SUPERCELL_MATRIX = np.diag([2, 2, 3])
REFERENCE_DISTANCE = 0.04
MLFCS_TO_THZ = np.sqrt(electron_volt / (angstrom**2 * atomic_mass)) / (2 * np.pi * 1e12)


def main() -> None:
    model = ForceConstants.load(ROOT / "fc-fit.mlfcs")
    primitive = model.space.primitive
    atoms = primitive.to_atoms()
    path = seekpath.get_explicit_k_path(
        (primitive.cell, primitive.scaled_positions, primitive.numbers),
        symprec=primitive.symprec,
        reference_distance=REFERENCE_DISTANCE,
    )
    seek_cell = np.asarray(path["primitive_lattice"])
    # seekpath labels are fractional in its standardized primitive reciprocal basis.
    # Convert those labels into the primitive basis stored in the MLFCS model.
    qpoints = np.asarray(path["explicit_kpoints_rel"]) @ (
        np.linalg.inv(seek_cell).T @ primitive.cell.T
    )
    harmonic = Harmonic(model)
    ours = harmonic.frequencies(qpoints)

    unitcell = PhonopyAtoms(
        numbers=atoms.numbers,
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )
    phonopy = Phonopy(
        unitcell,
        SUPERCELL_MATRIX,
        primitive_matrix="P",
        symprec=primitive.symprec,
        log_level=0,
    )
    actual = read(ROOT / "supercell.vasp")
    positions = np.asarray(phonopy.supercell.scaled_positions) - actual.get_scaled_positions()
    positions -= np.rint(positions)
    residual = float(np.max(np.linalg.norm(positions @ actual.cell.array, axis=1)))
    if (
        not np.array_equal(phonopy.supercell.numbers, actual.numbers)
        or residual >= primitive.symprec
    ):
        raise ValueError(
            "Phonopy supercell atom order does not match the exported FC2: "
            f"maximum position residual {residual:.6e} angstrom"
        )
    phonopy.force_constants = parse_FORCE_CONSTANTS(ROOT / "fc2-phonopy.txt")
    reference = phonopy.run_qpoints(qpoints).frequencies
    if reference.shape != ours.shape:
        raise RuntimeError(f"frequency shapes differ: {ours.shape} and {reference.shape}")

    difference = ours - reference
    # Phonopy and SciPy use slightly different physical-constant tables.  This
    # second metric removes that known scale difference from the algorithm check.
    same_units = ours * (phonopy.unit_conversion_factor / MLFCS_TO_THZ)
    adjusted_difference = same_units - reference
    away_from_gamma = np.asarray(path["explicit_kpoints_labels"]) != "GAMMA"
    stars = QStars.from_symmetry(QGrid.from_matrix(SUPERCELL_MATRIX), model.space.symmetry)
    mesh_ours = harmonic.frequencies(stars)
    mesh_reference = phonopy.run_qpoints(stars.points).frequencies
    mesh_difference = mesh_ours * (phonopy.unit_conversion_factor / MLFCS_TO_THZ) - mesh_reference
    worst = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
    result = {
        "material": "K4As4Pt2",
        "model": str((ROOT / "fc-fit.mlfcs").relative_to(ROOT)),
        "phonopy_fc2": str((ROOT / "fc2-phonopy.txt").relative_to(ROOT)),
        "bravais_lattice": path["bravais_lattice"],
        "path": [list(segment) for segment in path["path"]],
        "qpoint_count": len(qpoints),
        "bands": ours.shape[1],
        "mesh_qpoints": stars.grid.size,
        "mesh_irreducible_qpoints": len(stars.representatives),
        "mesh_star_weights": stars.weights.tolist(),
        "mesh_maximum_absolute_difference_same_units_thz": float(np.max(np.abs(mesh_difference))),
        "supercell_atom_order_residual_angstrom": residual,
        "mlfcs_conversion_thz": float(MLFCS_TO_THZ),
        "phonopy_conversion_thz": float(phonopy.unit_conversion_factor),
        "maximum_absolute_difference_thz": float(np.max(np.abs(difference))),
        "rms_difference_thz": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference_same_units_thz": float(np.max(np.abs(adjusted_difference))),
        "rms_difference_same_units_thz": float(np.sqrt(np.mean(adjusted_difference**2))),
        "maximum_absolute_difference_same_units_away_from_gamma_thz": float(
            np.max(np.abs(adjusted_difference[away_from_gamma]))
        ),
        "worst_qpoint": qpoints[worst[0]].tolist(),
        "worst_band_index": int(worst[1]),
        "worst_mlfcs_thz": float(ours[worst]),
        "worst_phonopy_thz": float(reference[worst]),
    }
    (HERE / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    distances = np.asarray(path["explicit_kpoints_linearcoord"])
    fig, (bands, errors) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for begin, end in path["explicit_segments"]:
        bands.plot(distances[begin:end], reference[begin:end], color="#34495e", lw=0.8)
        bands.plot(distances[begin:end], ours[begin:end], color="#e67e22", lw=0.6, ls="--")
        errors.plot(
            distances[begin:end],
            np.max(np.abs(difference[begin:end]), axis=1),
            color="#c0392b",
            lw=1.0,
        )
    labels = [
        (index, label) for index, label in enumerate(path["explicit_kpoints_labels"]) if label
    ]
    grouped: dict[float, list[str]] = {}
    for index, label in labels:
        position = float(distances[index])
        name = "Γ" if label == "GAMMA" else label.replace("_", "")
        if name not in grouped.setdefault(position, []):
            grouped[position].append(name)
    ticks = list(grouped)
    names = ["|".join(grouped[position]) for position in ticks]
    errors.set_xticks(ticks, names)
    for axis in (bands, errors):
        for position in ticks:
            axis.axvline(position, color="0.8", lw=0.5)
    bands.axhline(0.0, color="0.5", lw=0.6)
    bands.set_ylabel("Frequency (THz)")
    bands.legend(
        handles=[
            Line2D([], [], color="#34495e", label="Phonopy"),
            Line2D([], [], color="#e67e22", ls="--", label="MLFCS"),
        ],
        loc="upper right",
    )
    errors.set_ylabel("Max |Δ| (THz)")
    errors.set_xlabel("seekpath wave vector")
    fig.tight_layout()
    fig.savefig(HERE / "band-comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
