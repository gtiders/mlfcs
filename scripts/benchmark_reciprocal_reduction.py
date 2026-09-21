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
from ase.calculators.lj import LennardJones

from mlfcs import FiniteDifferenceCalculation, build_supercell
from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants
from mlfcs.reciprocal.fourier import dynamical_matrices, fourier_terms
from mlfcs.reciprocal.grid import irreducible_reciprocal_grid, reciprocal_quotient_grid
from mlfcs.reciprocal.sampling.harmonic import HarmonicSampler
from mlfcs.reciprocal.scph.fourier import harmonic_frequencies
from mlfcs.reciprocal.scph.solver import LoopSCPH
from mlfcs.reciprocal.statistics import OMEGA_TO_THZ, mode_sigma
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


def _benchmark_force_constants(
    primitive: Atoms, matrix: np.ndarray, cutoff: float
) -> tuple[ForceConstants, ForceConstants]:
    """Return ``(fc2, fc4)`` of a Lennard-Jones crystal in one reference supercell.

    FC2 comes from a finite-difference run, so it is symmetry-expanded by construction and
    passes the crystal-symmetry gate that every reciprocal consumer applies; FC4 is an
    on-site quartic tensor, which is only a source of quartic entries for the loop
    contraction and is not part of that check.
    """
    reference = build_supercell(primitive, matrix)
    # cutoff=-1 lets the library resolve the interaction radius from the reference, which is
    # what keeps a small reference identifiable instead of aliasing its own images.
    calculation = FiniteDifferenceCalculation(
        primitive, reference=reference, order=2, cutoff=-1, displacement=0.02
    )
    fc2 = calculation.run(LennardJones(epsilon=1.0, sigma=3.0, rc=9.0), acoustic_sum_rule=False)
    relation = fc2.relation
    quartic_sites = np.asarray(
        [[a, a, a, a] for a in range(len(relation.primitive))], dtype=np.int32
    )
    quartic_translations = np.zeros((len(relation.primitive), 3, 3), dtype=np.int32)
    quartic = np.zeros((len(relation.primitive), 3, 3, 3, 3), dtype=float)
    for site in range(len(relation.primitive)):
        quartic[site] = 0.01 * np.ones((3, 3, 3, 3))
    fc4 = SparseOrderForceConstants(4, quartic_sites, quartic_translations, quartic)
    return fc2, ForceConstants({}, relation.reference.copy(), sparse={4: fc4}, relation=relation)


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
    fc2, fc4 = _benchmark_force_constants(primitive, matrix, cutoff)
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

    # The sampler reads the per-atom cell/primitive metadata the structure relation puts on
    # its reference, so the relation's own reference is the right object here.
    sampler_reference = fc2.relation.reference
    sampler, sampler_matrices, sampler_cost = _measure(
        HarmonicSampler,
        primitive,
        sampler_reference,
        fc2.materialize(2),
        temperature=temperature,
        statistics="classical",
        # The benchmark model is a spring network, not an equilibrium crystal, so it has
        # imaginary modes; the sampler policy is explicit here exactly as in a real run.
        imaginary_modes="absolute",
    )
    snapshots = 4096
    _, _, sampling_cost = _measure(sampler.sample, snapshots, random_seed=20240921)
    _, free_energy_matrices, free_energy_cost = _measure(sampler.harmonic_free_energy)

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
        "| sampler quantity | irreducible | full grid |",
        "| --- | --- | --- |",
        (
            f"| sampler initialization diagonalizations | {int(sampler_matrices)} | "
            f"{sampler.grid.n_qpoints} |"
        ),
        f"| sampler init time (s) | {sampler_cost.seconds:.4f} | - |",
        f"| sampler init peak (MiB) | {sampler_cost.peak_mib:.2f} | - |",
        (
            f"| one batch of {snapshots} snapshots (s) | {sampling_cost.seconds:.4f} | - "
            "|"
        ),
        f"| harmonic free energy (s) | {free_energy_cost.seconds:.4f} | - |",
        f"| free energy diagonalizations | {int(free_energy_matrices)} | 0 |",
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
