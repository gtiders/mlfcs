"""Benchmark the irreducible reduction of the reciprocal paths.

The plan's acceptance criterion is that the diagonalization count equals ``N_irr`` and that
the full-grid expansion never re-enters an eigensolver, so this script reports counts and
timings for the same systems side by side with a full-grid reference that does the old
work.  Run it from the repository root::

    .venv/bin/python scripts/benchmark_reciprocal_reduction.py

The model is deliberately simple: an isotropic spring network for FC2 and an on-site
quartic tensor for FC4.  The point is the *bookkeeping*, not the physics, so the numbers are
comparable between the two paths and reproducible without a reference calculation.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list

from mlfcs import build_supercell
from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants
from mlfcs.reciprocal.fourier import dynamical_matrices, fourier_terms
from mlfcs.reciprocal.grid import irreducible_reciprocal_grid, reciprocal_quotient_grid
from mlfcs.reciprocal.scph.fourier import harmonic_frequencies
from mlfcs.reciprocal.scph.solver import LoopSCPH
from mlfcs.reciprocal.statistics import OMEGA_TO_THZ, mode_sigma
from mlfcs.structure.relation import StructureRelation
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations

SYSTEMS = {
    "hcp-2x2x2": ("Mg", np.diag((2, 2, 2)).astype(np.int64), 3.3),
    "diamond-2x2x2": ("Si", np.diag((2, 2, 2)).astype(np.int64), 2.6),
    "GaAs-3x2x2": ("GaAs", np.asarray([[3, 0, 0], [0, 2, 0], [0, 0, 2]]), 3.2),
}


@dataclass(frozen=True, slots=True)
class Measurement:
    """One measured quantity: value, wall time in seconds and peak memory in MiB."""

    value: float
    seconds: float
    peak_mib: float


def _primitive(name: str) -> Atoms:
    if name == "GaAs":
        return bulk("GaAs", "zincblende", a=5.653)
    if name == "Mg":
        return bulk("Mg", "hcp", a=3.2, c=5.2)
    return bulk("Si", "diamond", a=5.43)


def _spring_force_constants(primitive: Atoms, matrix: np.ndarray, cutoff: float) -> ForceConstants:
    """Return an isotropic spring-network FC2 plus an on-site quartic FC4."""
    reference = build_supercell(primitive, matrix)
    relation = StructureRelation.from_atoms(primitive, reference)
    first, second, shifts, distances = neighbor_list(
        "ijSd", relation.primitive, cutoff, self_interaction=True
    )
    blocks: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
    for atom_a, atom_b, shift, distance in zip(
        first.tolist(), second.tolist(), shifts.tolist(), distances.tolist(), strict=True
    ):
        if float(distance) >= cutoff or float(distance) == 0.0:
            continue
        key = (int(atom_a), int(atom_b), tuple(int(value) for value in shift))
        backward = (key[1], key[0], tuple(-value for value in key[2]))
        for entry in (key, backward):
            blocks[entry] = blocks.get(entry, 0.0) + np.eye(3) / len(relation.primitive)
    sites = np.asarray([[a, a] for a, b, _ in sorted(blocks)], dtype=np.int32)
    translations = np.asarray([[s] for _, _, s in sorted(blocks)], dtype=np.int32)
    tensors = np.asarray([blocks[key] for key in sorted(blocks)], dtype=float)
    fc2 = SparseOrderForceConstants(2, sites, translations, tensors)
    quartic_sites = np.asarray(
        [[a, a, a, a] for a in range(len(relation.primitive))], dtype=np.int32
    )
    quartic_translations = np.zeros((len(relation.primitive), 3, 3), dtype=np.int32)
    quartic = np.zeros((len(relation.primitive), 3, 3, 3, 3), dtype=float)
    for a in range(len(relation.primitive)):
        quartic[a] = 0.01 * np.ones((3, 3, 3, 3))
    fc4 = SparseOrderForceConstants(4, quartic_sites, quartic_translations, quartic)
    return (
        ForceConstants({}, reference, sparse={2: fc2}, relation=relation),
        ForceConstants({}, reference, sparse={4: fc4}, relation=relation),
    )


def _matrix_count(values: object) -> int:
    """Return how many matrices one eigensolver call diagonalizes."""
    array = np.asarray(values)
    return int(array.shape[0]) if array.ndim == 3 else 1


def _counted_diagonalizations(function, *args, **kwargs) -> tuple[object, int]:
    """Run ``function`` while counting the matrices its eigensolvers diagonalize.

    The count is the number of *matrices*, not the number of calls: a batched call over the
    representatives is one call and ``N_irr`` diagonalizations, which is the quantity the
    acceptance criterion is stated in.
    """
    calls = {"count": 0}
    original_values, original_vectors = np.linalg.eigvalsh, np.linalg.eigh

    def values_wrapper(values, *rest, **options):
        calls["count"] += _matrix_count(values)
        return original_values(values, *rest, **options)

    def vectors_wrapper(values, *rest, **options):
        calls["count"] += _matrix_count(values)
        return original_vectors(values, *rest, **options)

    np.linalg.eigvalsh, np.linalg.eigh = values_wrapper, vectors_wrapper
    try:
        result = function(*args, **kwargs)
    finally:
        np.linalg.eigvalsh, np.linalg.eigh = original_values, original_vectors
    return result, calls["count"]


def _measure(function, *args, **kwargs) -> tuple[object, int, Measurement]:
    """Return ``(result, eigensolver calls, timing and peak memory)``."""
    tracemalloc.start()
    start = time.perf_counter()
    result, calls = _counted_diagonalizations(function, *args, **kwargs)
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, calls, Measurement(float(calls), seconds, peak / 2**20)


def _full_grid_frequencies(fc2: ForceConstants, multiplier: int) -> np.ndarray:
    """Return the frequencies of every full q point, the pre-reduction reference."""
    relation = fc2.relation
    terms = fourier_terms(lattice_terms(fc2), relation.primitive)
    masses = np.asarray(relation.primitive.get_masses(), dtype=float)
    qpoints = reciprocal_quotient_grid(multiplier * relation.supercell_matrix).points
    values = np.linalg.eigvalsh(dynamical_matrices(terms, masses, qpoints))
    return np.sqrt(np.abs(values)) * np.sign(values) * OMEGA_TO_THZ


def lattice_terms(fc2: ForceConstants):
    """Return the primitive-lattice FC2 mapping of one force-constant object."""
    from mlfcs.force_constants.dense import lattice_fc2

    return lattice_fc2(fc2)


def _full_grid_covariance(fc2: ForceConstants, temperature: float) -> None:
    """Return nothing but do the pre-reduction covariance work, for timing only."""
    relation = fc2.relation
    terms = fourier_terms(lattice_terms(fc2), relation.primitive)
    masses = np.asarray(relation.primitive.get_masses(), dtype=float)
    qpoints = reciprocal_quotient_grid(relation.supercell_matrix).points
    values, vectors = np.linalg.eigh(dynamical_matrices(terms, masses, qpoints))
    sigma2 = (
        mode_sigma(values, temperature=temperature, statistics="classical") ** 2
    )
    return (vectors * sigma2[..., None, :]) @ vectors.conj().swapaxes(-1, -2)


def report(system: str, multiplier: int, temperature: float) -> str:
    """Return one markdown table for one system."""
    name, matrix, cutoff = SYSTEMS[system]
    primitive = _primitive(name)
    fc2, fc4 = _spring_force_constants(primitive, matrix, cutoff)
    symmetry = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=1e-5)
    mesh = harmonic_frequencies(fc2, multiplier)
    grid = irreducible_reciprocal_grid(multiplier * matrix, symmetry)
    weights = np.bincount(grid.weights.tolist())
    distribution = ", ".join(f"{size}x{count}" for size, count in enumerate(weights) if count)

    _, mesh_matrices, mesh_cost = _measure(harmonic_frequencies, fc2, multiplier)
    _, reference_matrices, reference_cost = _measure(_full_grid_frequencies, fc2, multiplier)

    solver = LoopSCPH(
        fc2=fc2,
        fc4=fc4,
        temperature=temperature,
        interpolation_multiplier=1,
        scph_multiplier=1,
        statistics="classical",
        max_iterations=1,
    )
    solver._run_single(temperature, None)  # warm the star decomposition cache
    sweep_mesh = solver._mesh(1)
    result, sweep_matrices, sweep_cost = _measure(solver._run_single, temperature, None)
    _, old_matrices, old_cost = _measure(_full_grid_covariance, fc2, temperature)

    lines = [
        f"### {system} (interpolation multiplier {multiplier})",
        "",
        "| quantity | irreducible | full grid |",
        "| --- | --- | --- |",
        f"| q points | {mesh.n_irreducible} | {mesh.n_qpoints} |",
        f"| reduction ratio | {mesh.reduction_ratio:.2f} | 1.00 |",
        f"| star size distribution | {distribution} | - |",
        f"| frequency diagonalizations | {int(mesh_matrices)} | {int(reference_matrices)} |",
        f"| frequency path time (s) | {mesh_cost.seconds:.4f} | {reference_cost.seconds:.4f} |",
        f"| frequency path peak (MiB) | {mesh_cost.peak_mib:.2f} | {reference_cost.peak_mib:.2f} |",
        (
            f"| one SCPH sweep diagonalizations | {int(sweep_matrices)} | "
            f"{3 * sweep_mesh.n_qpoints} |"
        ),
        f"| one SCPH sweep time (s) | {sweep_cost.seconds:.4f} | {old_cost.seconds:.4f} |",
        f"| one SCPH sweep peak (MiB) | {sweep_cost.peak_mib:.2f} | {old_cost.peak_mib:.2f} |",
        (
            f"| covariance diagonalizations (one eigh) | {sweep_mesh.n_irreducible} | "
            f"{int(old_matrices)} |"
        ),
        "",
        (
            f"SCPH converged: {result.converged}, iterations: {len(result.history)}, "
            f"expanded frequencies: {result.expand_frequencies().shape}"
        ),
        "",
        (
            "One SCPH sweep = one covariance plus the initial frequencies plus one "
            f"iteration, all on the reference grid ({sweep_mesh.n_qpoints} q points, "
            f"{sweep_mesh.n_irreducible} irreducible): {int(sweep_matrices)} diagonalizations "
            f"against {3 * sweep_mesh.n_qpoints} on the full grid.  The timings include the "
            "one-off symmetry analysis for the irreducible column, so small cells may look "
            "slower; the acceptance criterion is the diagonalization count, not the wall "
            "clock."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiplier", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--system", choices=sorted(SYSTEMS), action="append")
    options = parser.parse_args()
    selected = options.system or sorted(SYSTEMS)
    print("# Reciprocal reduction benchmark")
    print()
    print(
        "Diagonalizations count calls to `numpy.linalg.eigvalsh`/`eigh`; the full-grid "
        "column runs the pre-reduction computation for the same system."
    )
    print()
    for system in selected:
        print(report(system, options.multiplier, options.temperature))


if __name__ == "__main__":
    main()
