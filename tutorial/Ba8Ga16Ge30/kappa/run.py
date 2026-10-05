#!/usr/bin/env python3
"""Validate Ba8Ga16Ge30 ShengBTE IFC loading at the Gamma point with kALDo."""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ase.io import read, write
from kaldo.forceconstants import ForceConstants as KaldoForceConstants
from kaldo.phonons import Phonons

from mlfcs import ClusterMap, ForceConstants

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SUPERCELL_MATRIX = np.diag([2, 2, 2])


def export_kaldo_inputs(folder: Path) -> tuple[int, int]:
    """Export the fitted MLFCS model for kALDo's VASP/ShengBTE reader.

    FC2 is written in the VASP/phonopy ``FORCE_CONSTANTS_2ND`` text format;
    FC3 is written in ShengBTE ``FORCE_CONSTANTS_3RD`` format. The primitive
    POSCAR and both IFC files are staged in ``folder``. MLFCS exports Cartesian
    energy derivatives in eV and Å using the primitive atom order from
    ``primitive.vasp``.

    Args:
        folder: Directory for the temporary kALDo input files.

    Returns:
        A pair containing the primitive and supercell atom counts.

    Raises:
        FileNotFoundError: If the saved model or either structure file is
            missing.
        ValueError: If the saved model cannot be mapped to the configured
            supercell or lacks either required force-constant order.
    """
    model = ForceConstants.load(ROOT / "force_constants.mlfcs")
    primitive_atoms = read(ROOT / "primitive.vasp")
    supercell_atoms = read(ROOT / "supercell.vasp")
    mapping = ClusterMap(
        model.cluster_space,
        supercell_atoms,
        supercell_matrix=SUPERCELL_MATRIX,
    )
    folder.mkdir(parents=True, exist_ok=True)
    model.write(
        folder / "FORCE_CONSTANTS_2ND",
        mapping,
        format="phonopy",
        order=2,
        storage="text",
    )
    model.write(
        folder / "FORCE_CONSTANTS_3RD",
        mapping,
        format="shengbte",
        order=3,
    )
    write(folder / "POSCAR", primitive_atoms, format="vasp", direct=True, sort=False)
    return len(primitive_atoms), len(supercell_atoms)


def inspect_gamma_point() -> dict[str, object]:
    """Load both IFC orders and calculate harmonic frequencies at Gamma.

    This low-memory workflow validates the MLFCS-to-kALDo structure and file
    conventions. It does not calculate thermal conductivity: a single
    reciprocal-space point cannot replace the Brillouin-zone integration
    required for a bulk transport result.

    Returns:
        A JSON-serializable record with the Gamma-point frequencies in THz and
        the input/export metadata.

    Raises:
        ValueError: If kALDo returns a frequency array of the wrong shape or
            containing NaN or infinite values.
    """
    with tempfile.TemporaryDirectory(prefix="mlfcs-kaldo-ba-") as temporary:
        input_folder = Path(temporary)
        primitive_count, supercell_count = export_kaldo_inputs(input_folder)
        force_constants = KaldoForceConstants.from_folder(
            folder=str(input_folder),
            supercell=(2, 2, 2),
            format="vasp-sheng",
            third_energy_threshold=0.0,
        )
        phonons = Phonons(
            forceconstants=force_constants,
            kpts=(1, 1, 1),
            n_workers=1,
            folder=str(HERE / "kaldo-data"),
            storage="memory",
        )
        frequencies = np.asarray(phonons.frequency, dtype=np.float64)

    expected_shape = (1, 3 * primitive_count)
    if frequencies.shape != expected_shape:
        raise ValueError(f"unexpected Gamma-point frequency shape {frequencies.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Gamma-point frequencies contain NaN or infinite values")

    return {
        "material": "Ba8Ga16Ge30",
        "reciprocal_point": [0.0, 0.0, 0.0],
        "primitive_atoms": primitive_count,
        "supercell_atoms": supercell_count,
        "supercell_matrix": SUPERCELL_MATRIX.tolist(),
        "kaldo_version": version("kaldo"),
        "force_constants_format": "VASP/phonopy IFC2 + ShengBTE IFC3",
        "frequency_unit": "THz",
        "frequencies_THz": frequencies[0].tolist(),
        "frequency_min_THz": float(np.min(frequencies)),
        "frequency_max_THz": float(np.max(frequencies)),
        "thermal_conductivity_calculated": False,
    }


def main() -> None:
    """Run the Gamma-point IFC check and overwrite its log and JSON summary."""
    log_path = HERE / "gamma-point.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                report = inspect_gamma_point()
                (HERE / "gamma-point.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    "kALDo Gamma-point IFC check passed: "
                    f"{report['primitive_atoms']} primitive atoms, "
                    f"{len(report['frequencies_THz'])} frequencies"
                )
        except BaseException:
            traceback.print_exc(file=log_file)
            raise


if __name__ == "__main__":
    main()
