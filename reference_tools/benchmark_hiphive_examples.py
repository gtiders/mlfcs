"""Reproduce the public hiPhive Si and BaGaGe comparison cases.

The external ``hiphive-examples`` checkout is intentionally not vendored.  See
``tests/reference/hiphive/README.md`` for the exact checkout procedure.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import jax
import numpy as np
import spglib
from ase import Atoms
from ase.db import connect
from ase.geometry import find_mic
from ase.io import read
from scipy import sparse

SI_CUTOFFS = {2: 9.65, 3: 9.65}
BGG_DATABASES = (
    "mc_rattle_std0.042_vdW-DF-cx.db",
    "mc_rattle_based_md_T300_vdW-DF-cx.db",
    "mc_rattle_based_md_T650_vdW-DF-cx.db",
)
BGG_CUTOFFS = {2: 5.4, 3: 4.35, 4: 4.35}
BGG_MAX_BODY_ORDERS = {2: 2, 3: 2, 4: 2}


def _si_paths(root: Path) -> tuple[Path, Path]:
    base = root / "examples/Si_bulk/dft_calculations/structures"
    return base / "POSCAR_ideal_size5", base / "silicon-bulk-dft-n5-rattled-0.03.db"


def _si_primitive(reference: Atoms) -> Atoms:
    """Recover hiPhive's primitive while retaining the dataset's origin."""
    cell = spglib.find_primitive(
        (reference.cell, reference.get_scaled_positions(), reference.numbers), symprec=1e-5
    )
    if cell is None:
        raise RuntimeError("spglib could not find the Si primitive cell")
    primitive_cell = np.asarray(cell[0])
    fractional = (reference.positions @ np.linalg.inv(primitive_cell)) % 1.0
    unique: list[np.ndarray] = []
    for position in fractional:
        if not any(
            np.linalg.norm((position - previous) - np.rint(position - previous)) < 1e-5
            for previous in unique
        ):
            unique.append(position)
    if len(unique) != 2:
        raise RuntimeError(f"expected a two-atom Si primitive, found {len(unique)} atoms")
    return Atoms("Si2", scaled_positions=unique, cell=primitive_cell, pbc=True)


def run_si_mlfcs(root: Path, output: Path) -> None:
    from mlfcs.fitting import ForceConstantFitter

    reference_path, database_path = _si_paths(root)
    reference = read(reference_path)
    structures = [row.toatoms() for row in connect(database_path).select()]
    fitter = ForceConstantFitter(
        _si_primitive(reference),
        reference,
        orders=(2, 3),
        cutoffs=SI_CUTOFFS,
        symprec=1e-5,
        jax_platform="cpu",
    )
    result = fitter.fit(structures, validation_split=0.0, acoustic_sum_rule=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "mlfcs_si_fc23.npz",
        parameters=result.parameters,
        covariance=result.covariance,
        training_force_rmse=result.diagnostics.training_force_rmse,
        training_relative_force_error=result.diagnostics.training_relative_force_error,
    )


def run_si_hiphive(root: Path, output: Path) -> None:
    from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
    from hiphive.utilities import get_displacements
    from trainstation import Optimizer

    reference_path, database_path = _si_paths(root)
    reference = read(reference_path)
    cluster_space = ClusterSpace(reference, [9.65, 9.65], symprec=1e-5)
    container = StructureContainer(cluster_space)
    for row in connect(database_path).select():
        atoms = row.toatoms()
        forces = atoms.get_forces()
        displacements = get_displacements(atoms, reference)
        atoms.positions = reference.positions
        atoms.calc = None
        atoms.new_array("forces", forces)
        atoms.new_array("displacements", displacements)
        container.add_structure(atoms)
    design, target = container.get_fit_data()
    optimizer = Optimizer(
        (design, target),
        fit_method="least-squares",
        train_size=1.0,
        standardize=True,
        seed=42,
    )
    optimizer.train()
    residual = design @ optimizer.parameters - target
    output.mkdir(parents=True, exist_ok=True)
    ForceConstantPotential(cluster_space, optimizer.parameters).write(
        str(output / "hiphive_si_fc23.fcp")
    )
    print(f"hiPhive force RMSE: {np.sqrt(np.mean(residual**2)):.12e} eV/angstrom")
    print(f"hiPhive relative force error: {np.linalg.norm(residual) / np.linalg.norm(target):.12e}")


def compare_si_force_constants(root: Path, output: Path) -> None:
    """Compare tensors after aligning supercell atoms and IFC axes."""
    from hiphive import ForceConstantPotential

    from mlfcs.fitting import ForceConstantFitter
    from mlfcs.fitting.constraints import build_wick_to_taylor_transform
    from mlfcs.fitting.parameterization import expand_sparse

    reference = read(_si_paths(root)[0])
    fitter = ForceConstantFitter(
        _si_primitive(reference),
        reference,
        orders=(2, 3),
        cutoffs=SI_CUTOFFS,
        symprec=1e-5,
        jax_platform="cpu",
        verbose=False,
    )
    state = np.load(output / "mlfcs_si_fc23.npz")
    transform = build_wick_to_taylor_transform(fitter.calculations, state["covariance"])
    sparse = expand_sparse(transform @ state["parameters"], fitter.calculations, 2, len(reference))
    mapping = []
    for number, position in zip(
        fitter.canonical_supercell.numbers,
        fitter.canonical_supercell.positions,
        strict=True,
    ):
        candidates = np.flatnonzero(reference.numbers == number)
        _, distances = find_mic(
            reference.positions[candidates] - position, reference.cell, pbc=True
        )
        mapping.append(int(candidates[np.argmin(distances)]))
    if len(set(mapping)) != len(reference):
        raise RuntimeError("Si supercell atom mapping is not one-to-one")
    reference_fcs = (
        ForceConstantPotential.read(str(output / "hiphive_si_fc23.fcp"))
        .get_force_constants(reference)
        ._fc_dict
    )
    for order, values in sparse.items():
        differences = []
        references = []
        for cluster, tensor in zip(values.clusters, values.tensors, strict=True):
            mapped = np.asarray([mapping[int(atom)] for atom in cluster])
            permutation = np.argsort(mapped, kind="stable")
            key = tuple(mapped[permutation])
            differences.append(np.transpose(tensor, tuple(permutation)) - reference_fcs[key])
            references.append(reference_fcs[key])
        difference = np.asarray(differences)
        reference_values = np.asarray(references)
        print(
            f"FC{order}: max_abs={np.max(np.abs(difference)):.12e}, "
            f"rmse={np.sqrt(np.mean(difference**2)):.12e}, "
            f"relative_rms={np.linalg.norm(difference) / np.linalg.norm(reference_values):.12e}"
        )


def diagnose_bagage_wick(root: Path, *, samples: int, seed: int) -> None:
    """Measure sampled linear/cubic feature correlations before and after Wick."""
    base = root / "examples/BaGaGe_clathrate/dft_calculations/structures"
    reference = read(base / "POSCAR_groundstate_vdW-DF-cx")
    displacements = []
    for name in BGG_DATABASES:
        for row in connect(base / name).select():
            value, _ = find_mic(
                row.toatoms().positions - reference.positions, reference.cell, pbc=True
            )
            displacements.append(value.reshape(-1))
    displacement = np.asarray(displacements)
    covariance = displacement.T @ displacement / len(displacement)
    triples = np.random.default_rng(seed).integers(0, displacement.shape[1], size=(samples, 3))
    left, middle, right = triples.T
    taylor = displacement[:, left] * displacement[:, middle] * displacement[:, right]
    wick = (
        taylor
        - displacement[:, left] * covariance[middle, right]
        - displacement[:, middle] * covariance[left, right]
        - displacement[:, right] * covariance[left, middle]
    )
    for label, features in (("Taylor cubic", taylor), ("Wick cubic", wick)):
        linear = displacement - displacement.mean(axis=0)
        cubic = features - features.mean(axis=0)
        correlations = np.abs(
            (linear.T @ cubic)
            / (np.linalg.norm(linear, axis=0)[:, None] * np.linalg.norm(cubic, axis=0)[None])
        ).ravel()
        print(
            f"{label}: mean={correlations.mean():.8f}, "
            f"rms={np.sqrt(np.mean(correlations**2)):.8f}, "
            f"p95={np.quantile(correlations, 0.95):.8f}, "
            f"p99={np.quantile(correlations, 0.99):.8f}, max={correlations.max():.8f}"
        )


def _bagage_data(root: Path) -> tuple[Atoms, list[Atoms]]:
    """Load the exact public 200-snapshot BaGaGe fitting dataset."""
    base = root / "examples/BaGaGe_clathrate/dft_calculations/structures"
    reference = read(base / "POSCAR_groundstate_vdW-DF-cx")
    structures = [row.toatoms() for name in BGG_DATABASES for row in connect(base / name).select()]
    if len(structures) != 200:
        raise RuntimeError(f"expected 200 BaGaGe snapshots, found {len(structures)}")
    return reference, structures


def _save_sparse_force_constants(output: Path, result) -> None:
    payload: dict[str, np.ndarray] = {
        "parameters": result.parameters,
        "parameter_scale": result.parameter_scale,
        "covariance": result.covariance,
    }
    for order, values in result.force_constants.sparse.items():
        payload[f"fc{order}_clusters"] = values.clusters
        payload[f"fc{order}_tensors"] = values.tensors
    np.savez_compressed(output, **payload)


def _diagnostics_payload(result) -> dict[str, object]:
    diagnostics = result.diagnostics
    return {
        "training_force_rmse_eV_per_angstrom": diagnostics.training_force_rmse,
        "validation_force_rmse_eV_per_angstrom": diagnostics.validation_force_rmse,
        "training_relative_force_error": diagnostics.training_relative_force_error,
        "validation_relative_force_error": diagnostics.validation_relative_force_error,
        "maximum_constraint_residual": diagnostics.maximum_constraint_residual,
        "iterations": diagnostics.iterations,
        "stop_code": diagnostics.stop_code,
        "reduced_design_kernel_signatures": diagnostics.design_kernel_signatures,
        "design_tiles": diagnostics.design_tiles,
        "static_device_bytes": diagnostics.static_device_bytes,
        "gram_feature_passes": diagnostics.gram_feature_passes,
        "prediction_feature_passes": diagnostics.prediction_feature_passes,
    }


def run_bagage_mlfcs(root: Path, output: Path, *, validation_split: float, seed: int) -> None:
    """Fit the published two-body FC2+FC3+FC4 BaGaGe model with MLFCS."""
    from mlfcs.fitting import ForceConstantFitter

    reference, structures = _bagage_data(root)
    fitter = ForceConstantFitter(
        reference,
        reference,
        orders=(2, 3, 4),
        cutoffs=BGG_CUTOFFS,
        max_body_orders=BGG_MAX_BODY_ORDERS,
        symprec=1e-4,
        jax_platform="cpu",
    )
    result = fitter.fit(
        structures,
        validation_split=validation_split,
        seed=seed,
        acoustic_sum_rule=True,
        tolerance=1e-8,
        max_iterations=10_000,
    )
    output.mkdir(parents=True, exist_ok=True)
    _save_sparse_force_constants(output / "mlfcs_bagage_fc234.npz", result)
    metadata = _diagnostics_payload(result)
    metadata.update(
        n_structures=len(structures),
        validation_split=validation_split,
        cutoffs_angstrom=BGG_CUTOFFS,
        max_body_orders=BGG_MAX_BODY_ORDERS,
        unconstrained_parameters=fitter.n_parameters,
    )
    (output / "mlfcs_bagage_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def run_bagage_hiphive(root: Path, output: Path, *, validation_split: float, seed: int) -> None:
    """Fit the same public model with the installed hiPhive implementation."""
    from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
    from hiphive.cutoffs import Cutoffs
    from hiphive.utilities import get_displacements
    from trainstation import Optimizer

    reference, structures = _bagage_data(root)
    # The outer list is essential: one cutoff row means two-body support for
    # every included IFC order, exactly as in the public hiPhive example.
    cluster_space = ClusterSpace(
        reference,
        Cutoffs([[BGG_CUTOFFS[2], BGG_CUTOFFS[3], BGG_CUTOFFS[4]]]),
        symprec=1e-4,
    )
    container = StructureContainer(cluster_space)
    for atoms in structures:
        forces = atoms.get_forces()
        displacements = get_displacements(atoms, reference)
        atoms.positions = reference.positions
        atoms.calc = None
        atoms.new_array("forces", forces)
        atoms.new_array("displacements", displacements)
        container.add_structure(atoms)
    design, target = container.get_fit_data()
    optimizer = Optimizer(
        (design, target),
        fit_method="least-squares",
        train_size=1.0 - validation_split,
        standardize=True,
        seed=seed,
    )
    optimizer.train()
    prediction = design @ optimizer.parameters
    residual = prediction - target
    output.mkdir(parents=True, exist_ok=True)
    potential = ForceConstantPotential(cluster_space, optimizer.parameters)
    potential.write(str(output / "hiphive_bagage_fc234.fcp"))
    summary = {
        "force_rmse_eV_per_angstrom": float(np.sqrt(np.mean(residual**2))),
        "relative_force_error": float(np.linalg.norm(residual) / np.linalg.norm(target)),
        "n_equations": len(target),
        "n_parameters": len(optimizer.parameters),
        "n_structures": len(structures),
        "validation_split": validation_split,
        "cutoffs_angstrom": BGG_CUTOFFS,
        "max_body_order": 2,
    }
    (output / "hiphive_bagage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def compare_bagage_force_constants(root: Path, output: Path) -> None:
    """Compare the two fits after explicit atom and IFC-axis alignment."""
    from hiphive import ForceConstantPotential

    from mlfcs.fitting import ForceConstantFitter

    reference, _ = _bagage_data(root)
    fitter = ForceConstantFitter(
        reference,
        reference,
        orders=(2, 3, 4),
        cutoffs=BGG_CUTOFFS,
        max_body_orders=BGG_MAX_BODY_ORDERS,
        symprec=1e-4,
        jax_platform="cpu",
        verbose=False,
    )
    saved = np.load(output / "mlfcs_bagage_fc234.npz")
    mapping = []
    for number, position in zip(
        fitter.canonical_supercell.numbers, fitter.canonical_supercell.positions, strict=True
    ):
        candidates = np.flatnonzero(reference.numbers == number)
        _, distances = find_mic(
            reference.positions[candidates] - position, reference.cell, pbc=True
        )
        mapping.append(int(candidates[np.argmin(distances)]))
    if len(set(mapping)) != len(reference):
        raise RuntimeError("BaGaGe supercell atom mapping is not one-to-one")
    reference_fcs = (
        ForceConstantPotential.read(str(output / "hiphive_bagage_fc234.fcp"))
        .get_force_constants(reference)
        ._fc_dict
    )
    rows: dict[str, dict[str, float | int]] = {}
    for order in (2, 3, 4):
        clusters = saved[f"fc{order}_clusters"]
        tensors = saved[f"fc{order}_tensors"]
        differences = []
        references = []
        missing = 0
        for cluster, tensor in zip(clusters, tensors, strict=True):
            mapped = np.asarray([mapping[int(atom)] for atom in cluster])
            permutation = np.argsort(mapped, kind="stable")
            key = tuple(mapped[permutation])
            expected = reference_fcs.get(key)
            if expected is None:
                missing += 1
                continue
            differences.append(np.transpose(tensor, tuple(permutation)) - expected)
            references.append(expected)
        difference = np.asarray(differences)
        expected = np.asarray(references)
        rows[f"fc{order}"] = {
            "matched_clusters": len(differences),
            "missing_clusters": missing,
            "max_abs": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "relative_rms": float(np.linalg.norm(difference) / np.linalg.norm(expected)),
        }
    (output / "bagage_tensor_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


def _bagage_reduced_operator(root: Path):
    """Build the identical ASR-reduced physical-design state used by the fit."""
    from mlfcs.fitting import ForceConstantFitter
    from mlfcs.fitting.basis import symmetrized_covariance
    from mlfcs.fitting.constraints import build_joint_constraints
    from mlfcs.fitting.data import FitDataset
    from mlfcs.fitting.design import ForceDesignOperator
    from mlfcs.fitting.solver import explicit_constraint_null_space

    reference, structures = _bagage_data(root)
    fitter = ForceConstantFitter(
        reference,
        reference,
        orders=(2, 3, 4),
        cutoffs=BGG_CUTOFFS,
        max_body_orders=BGG_MAX_BODY_ORDERS,
        symprec=1e-4,
        jax_platform="cpu",
        verbose=False,
    )
    dataset = FitDataset.from_atoms(fitter.geometry, structures)
    permutation = fitter.geometry.internal_permutation
    displacements = dataset.displacements[:, permutation]
    forces = dataset.forces[:, permutation].reshape(-1)
    covariance = symmetrized_covariance(displacements, fitter.calculations[0])
    constraints = build_joint_constraints(fitter.calculations, acoustic=True, rotational_mode=0)
    parameter_map = explicit_constraint_null_space(constraints.matrix, tolerance=1e-11)
    operator = ForceDesignOperator(
        displacements,
        covariance,
        fitter.order_tensors,
        fitter.n_parameters,
        batch_size=1,
        parameter_map=parameter_map,
        device=fitter.jax_device,
    )
    counts = [
        sum(orbit.dimension for orbit in calculation.orbit_space.orbits)
        for calculation in fitter.calculations
    ]
    return fitter, operator, parameter_map, forces, counts


def _reduced_order_columns(parameter_map: np.ndarray, counts: list[int]) -> dict[int, np.ndarray]:
    """Identify the ASR-reduced coordinates carrying each uncoupled IFC order."""
    offsets = np.cumsum([0, *counts])
    result = {}
    for order, (begin, end) in enumerate(pairwise(offsets), start=2):
        block = sparse.csc_matrix(parameter_map[begin:end])
        result[order] = np.flatnonzero(np.diff(block.indptr) > 0)
    if np.intersect1d(result[2], result[4]).size:
        raise RuntimeError("ASR parameter map unexpectedly mixes FC2 and FC4 coordinates")
    return result


def _fc2_fc4_gram_blocks(operator, parameter_map, columns, *, taylor: bool):
    """Stream normalized-design Gram blocks without retaining a dense global A."""
    from mlfcs.fitting.design import prepare_design_kernel_groups

    n2, n4 = len(columns[2]), len(columns[4])
    g22 = np.zeros((n2, n2))
    g24 = np.zeros((n2, n4))
    g44 = np.zeros((n4, n4))
    builders, batch_size = prepare_design_kernel_groups(operator)
    rows_per_structure = int(np.prod(operator.force_shape[1:]))
    covariance = (
        jax.device_put(np.zeros_like(np.asarray(operator.covariance)), operator.program.device)
        if taylor
        else operator.covariance
    )
    mapping = sparse.csc_matrix(parameter_map)
    for begin in range(0, len(operator.displacements), batch_size):
        end = min(begin + batch_size, len(operator.displacements))
        rows = (end - begin) * rows_per_structure
        design = np.zeros((rows, operator.n_parameters))
        displacement = jax.device_put(operator.displacements[begin:end], operator.program.device)
        for group in builders:
            contributions = np.asarray(
                group.kernel(displacement, covariance, *group.device_arguments)
            )
            for contribution, tile_columns in zip(contributions, group.columns, strict=True):
                design[:, tile_columns] += contribution.reshape(rows, -1)
        reduced = np.asarray(mapping.T @ design.T).T
        fc2 = reduced[:, columns[2]]
        fc4 = reduced[:, columns[4]]
        g22 += fc2.T @ fc2
        g24 += fc2.T @ fc4
        g44 += fc4.T @ fc4
    return g22, g24, g44


def _cross_block_metrics(g22, g24, g44) -> dict[str, float | int]:
    """Report pairwise and subspace FC2--FC4 collinearity after ASR."""
    norm2 = np.sqrt(np.maximum(np.diag(g22), np.finfo(float).tiny))
    norm4 = np.sqrt(np.maximum(np.diag(g44), np.finfo(float).tiny))
    pairwise = g24 / norm2[:, None] / norm4[None, :]

    def whitening(gram):
        values, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
        keep = values > max(float(values.max()), 1.0) * 1e-10
        return vectors[:, keep] / np.sqrt(values[keep]), int(np.count_nonzero(keep))

    left, rank2 = whitening(g22)
    right, rank4 = whitening(g44)
    canonical = np.linalg.svd(left.T @ g24 @ right, compute_uv=False)
    joint = np.block([[g22, g24], [g24.T, g44]])
    scales = np.concatenate([norm2, norm4])
    joint = joint / scales[:, None] / scales[None, :]
    values = np.linalg.eigvalsh((joint + joint.T) * 0.5)
    active = values[values > max(float(values.max()), 1.0) * 1e-10]
    return {
        "fc2_rank": rank2,
        "fc4_rank": rank4,
        "maximum_pairwise_normalized_correlation": float(np.max(np.abs(pairwise))),
        "rms_pairwise_normalized_correlation": float(np.sqrt(np.mean(pairwise**2))),
        "maximum_canonical_correlation": float(canonical[0]) if len(canonical) else 0.0,
        "joint_active_condition_number": float(active.max() / active.min()),
    }


def diagnose_bagage_fc2_fc4_collinearity(root: Path, output: Path) -> None:
    """Compare full ASR-reduced FC2--FC4 Gram coupling in Taylor and Wick bases."""
    _fitter, operator, parameter_map, _forces, counts = _bagage_reduced_operator(root)
    columns = _reduced_order_columns(parameter_map, counts)
    results = {}
    for label, taylor in (("taylor", True), ("wick", False)):
        started = perf_counter()
        blocks = _fc2_fc4_gram_blocks(operator, parameter_map, columns, taylor=taylor)
        metrics = _cross_block_metrics(*blocks)
        metrics["gram_block_wall_seconds"] = perf_counter() - started
        results[label] = metrics
    output.mkdir(parents=True, exist_ok=True)
    (output / "bagage_fc2_fc4_collinearity.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def benchmark_bagage_gram_cache(root: Path, output: Path) -> None:
    """Measure cold streamed Gram construction and a genuine warm cache hit."""
    from mlfcs.fitting.model import _StreamingGramSystem

    _fitter, operator, _parameter_map, forces, _counts = _bagage_reduced_operator(root)
    original = Path.cwd()
    with TemporaryDirectory(prefix="mlfcs-bagage-gram-") as temporary:
        os.chdir(temporary)
        try:
            started = perf_counter()
            _StreamingGramSystem.from_operator(operator, forces)
            cold = perf_counter() - started
            started = perf_counter()
            _StreamingGramSystem.from_operator(operator, forces)
            warm = perf_counter() - started
        finally:
            os.chdir(original)
    result = {"cold_seconds": cold, "warm_seconds": warm, "speedup": cold / warm}
    output.mkdir(parents=True, exist_ok=True)
    (output / "bagage_gram_cache.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "si-mlfcs",
            "si-hiphive",
            "si-compare",
            "bagage-wick",
            "bagage-mlfcs",
            "bagage-hiphive",
            "bagage-compare",
            "bagage-collinearity",
            "bagage-gram-cache",
        ),
    )
    parser.add_argument("--examples", type=Path, default=Path("hiphive-examples"))
    parser.add_argument("--output", type=Path, default=Path("results/hiphive_examples"))
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-split", type=float, default=0.1)
    args = parser.parse_args()
    if args.mode == "si-mlfcs":
        run_si_mlfcs(args.examples, args.output)
    elif args.mode == "si-hiphive":
        run_si_hiphive(args.examples, args.output)
    elif args.mode == "si-compare":
        compare_si_force_constants(args.examples, args.output)
    elif args.mode == "bagage-wick":
        diagnose_bagage_wick(args.examples, samples=args.samples, seed=args.seed)
    elif args.mode == "bagage-mlfcs":
        run_bagage_mlfcs(
            args.examples, args.output, validation_split=args.validation_split, seed=args.seed
        )
    elif args.mode == "bagage-hiphive":
        run_bagage_hiphive(
            args.examples, args.output, validation_split=args.validation_split, seed=args.seed
        )
    elif args.mode == "bagage-compare":
        compare_bagage_force_constants(args.examples, args.output)
    elif args.mode == "bagage-collinearity":
        diagnose_bagage_fc2_fc4_collinearity(args.examples, args.output)
    else:
        benchmark_bagage_gram_cache(args.examples, args.output)


if __name__ == "__main__":
    main()
