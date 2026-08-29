"""Compare KCl FC2 fitting with and without periodic FC2 completion."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from ase import Atoms
from ase.io import read
from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS, read_force_constants_hdf5, write_FORCE_CONSTANTS
from phonopy.interface.calculator import read_crystal_structure
from phonopy.structure.atoms import PhonopyAtoms
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

from mlfcs import write_force_constants
from mlfcs.fitting import ForceConstantFitter

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "fit.log"
MODEL = ROOT.parent / "input" / "polymlp.yaml"
SUPERCELL_MATRIX = np.diag((2, 2, 2))


class _Tee:
    """Mirror stdout and stderr to the terminal and this task's fit.log."""

    def __init__(self, terminal, log_file) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()


def _fit(periodic: bool, structures, primitive, reference):
    label = "periodic" if periodic else "exact"
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 6.0},
        max_body_orders={2: 2},
        periodic_fc2_completion=periodic,
    )
    result = fitter.fit(
        structures,
        validation_split=0.0,
        acoustic_sum_rule=True,
        cache_directory=ROOT / f"fit-cache-{label}",
    )
    write_force_constants(result.force_constants, ROOT / f"fit-{label}.h5", format="hdf5")
    write_force_constants(
        result.force_constants,
        ROOT / f"FORCE_CONSTANTS_FIT_{label.upper()}",
        format="phonopy",
        order=2,
    )
    write_force_constants(
        result.force_constants,
        ROOT / f"force_constants-{label}.hdf5",
        format="phonopy_hdf5",
        order=2,
    )
    return result


def _phonon(path: Path, force_constants: Path, *, text_format: bool = False) -> tuple[dict, object]:
    cell, _ = read_crystal_structure(filename=str(path), interface_mode="vasp")
    if cell is None:
        raise ValueError(f"phonopy could not read {path}")
    phonon = Phonopy(cell, np.eye(3, dtype=int), primitive_matrix="auto")
    if text_format:
        phonon.force_constants = parse_FORCE_CONSTANTS(filename=str(force_constants))
    else:
        phonon.force_constants = read_force_constants_hdf5(str(force_constants))
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
    points = path_data["point_coords"]
    paths = [np.linspace(points[start], points[end], 101) for start, end in segments]
    connections = [
        index + 1 < len(segments) and end == segments[index + 1][0]
        for index, (_, end) in enumerate(segments)
    ]
    phonon.run_band_structure(paths, path_connections=connections)
    return path_data, phonon.band_structure


def _phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def _ase_atoms(atoms: PhonopyAtoms) -> Atoms:
    return Atoms(
        numbers=atoms.numbers,
        cell=atoms.cell,
        scaled_positions=atoms.scaled_positions,
        pbc=True,
    )


def _phonopy_with_polymlp(path: Path) -> tuple[dict, object]:
    """Generate FC2 with Phonopy displacements and the PolyMLP calculator."""
    unitcell = read(path)
    reference = read(ROOT / "supercell.vasp")
    phonon = Phonopy(_phonopy_atoms(unitcell), SUPERCELL_MATRIX, primitive_matrix="auto")
    phonopy_reference = _ase_atoms(phonon.supercell)
    if not np.array_equal(reference.numbers, phonopy_reference.numbers):
        raise RuntimeError("phonopy and reference supercell atom orders differ")
    if not np.allclose(reference.cell.array, phonopy_reference.cell.array, atol=1e-8):
        raise RuntimeError("phonopy and reference supercell cells differ")

    phonon.generate_displacements(distance=0.01)
    calculator = PolymlpASECalculator(pot=MODEL)
    forces = []
    for displaced in phonon.supercells_with_displacements:
        structure = _ase_atoms(displaced)
        structure.calc = calculator
        forces.append(structure.get_forces())
    phonon.forces = np.asarray(forces)
    phonon.produce_force_constants()
    write_FORCE_CONSTANTS(
        phonon.force_constants,
        filename=ROOT / "FORCE_CONSTANTS_POLYMLP_PHONOPY",
    )
    return _band_structure(phonon)


def _band_structure(phonon: Phonopy) -> tuple[dict, object]:
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
    points = path_data["point_coords"]
    paths = [np.linspace(points[start], points[end], 101) for start, end in segments]
    connections = [
        index + 1 < len(segments) and end == segments[index + 1][0]
        for index, (_, end) in enumerate(segments)
    ]
    # NAC is intentionally disabled: no nac_params are assigned.
    phonon.run_band_structure(paths, path_connections=connections)
    return path_data, phonon.band_structure


def _plot(path_data: dict, bands: dict[str, object]) -> None:
    segments = path_data["path"]
    ticks: dict[float, str] = {}
    for (start, end), distance in zip(segments, bands["exact"].distances, strict=True):
        for location, label in ((float(distance[0]), start), (float(distance[-1]), end)):
            label = "Γ" if label == "GAMMA" else label
            if location in ticks and ticks[location] != label:
                ticks[location] += f"|{label}"
            else:
                ticks[location] = label

    figure, axis = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    colors = {"exact": "#176b87", "periodic": "#d97706", "polymlp": "#111827"}
    labels = {
        "exact": "Exact-R FC2 (HDF5)",
        "periodic": "Periodic FC2 completion (HDF5)",
        "polymlp": "PolyMLP forces + Phonopy",
    }
    styles = {"exact": "-", "periodic": "-", "polymlp": "--"}
    for name, band in bands.items():
        first_branch = True
        for distance, frequencies in zip(band.distances, band.frequencies, strict=True):
            for branch in np.asarray(frequencies).T:
                axis.plot(
                    distance,
                    branch,
                    color=colors[name],
                    linestyle=styles[name],
                    linewidth=1.25 if name != "polymlp" else 1.0,
                    alpha=0.9,
                    label=labels[name] if first_branch else None,
                )
                first_branch = False
    for location in ticks:
        axis.axvline(location, color="#cbd5e1", linewidth=0.6, zorder=0)
    axis.axhline(0.0, color="#64748b", linewidth=0.6, linestyle="--")
    axis.set_xticks(list(ticks), list(ticks.values()))
    axis.set_ylabel("Frequency (THz)")
    axis.set_xlabel("Seekpath high-symmetry path")
    axis.set_title("KCl FC2 comparison: fitted FC2 and direct PolyMLP, NAC off")
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.5)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    figure.savefig(ROOT / "periodic-fc2-comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _run() -> None:
    primitive = read(ROOT / "primitive.vasp")
    reference = read(ROOT / "supercell.vasp")
    structures = read(ROOT / "train.extxyz", index=":")
    exact = _fit(False, structures, primitive, reference)
    periodic = _fit(True, structures, primitive, reference)
    path_data, exact_band = _phonon(ROOT / "supercell.vasp", ROOT / "force_constants-exact.hdf5")
    periodic_path, periodic_band = _phonon(
        ROOT / "supercell.vasp", ROOT / "force_constants-periodic.hdf5"
    )
    if periodic_path["path"] != path_data["path"]:
        raise RuntimeError("seekpath returned different high-symmetry paths")
    polymlp_path, polymlp_band = _phonopy_with_polymlp(ROOT / "primitive.vasp")
    if polymlp_path["path"] != path_data["path"]:
        raise RuntimeError("seekpath returned different high-symmetry paths for PolyMLP FC2")
    _plot(
        path_data,
        {
            "exact": exact_band,
            "periodic": periodic_band,
            "polymlp": polymlp_band,
        },
    )
    summary = {
        "snapshots": len(structures),
        "periodic_fc2_completion": True,
        "nac": False,
        "polymlp_phonopy_force_constants": "FORCE_CONSTANTS_POLYMLP_PHONOPY",
        "path": [list(segment) for segment in path_data["path"]],
        "exact_training_relative_force_error": exact.training_relative_force_error,
        "periodic_training_relative_force_error": periodic.training_relative_force_error,
    }
    (ROOT / "periodic-fc2-comparison.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "wrote exact/periodic FC2 files, periodic-fc2-comparison.png, "
        "and periodic-fc2-comparison.json"
    )


def main() -> None:
    with LOG.open("w", encoding="utf-8") as log_file:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log_file), _Tee(stderr, log_file)
        package_logger = logging.getLogger("mlfcs")
        handler = logging.StreamHandler(log_file)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        package_logger.addHandler(handler)
        try:
            _run()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            package_logger.removeHandler(handler)
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
