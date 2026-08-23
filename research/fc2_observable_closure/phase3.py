#!/usr/bin/env python3
"""Structure, robustness, and architecture study for FC2 observable closure."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from phonopy import load
from scipy.linalg import subspace_angles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASE = ROOT / "examples" / "sscha" / "KCl"
RESULTS = HERE / "results-phase3.json"
for path in (HERE, CASE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import prototype as phase2
from common import HARMONIC_PATH, POTENTIAL_PATH, ase_from_phonopy

from mlfcs.constraints.translational import build_translational_constraints
from mlfcs.fitting.linear_solvers import explicit_constraint_null_space
from mlfcs.force_constants.dense import compact_fc2
from mlfcs.force_constants.expansion import expand_primitive_parameters
from mlfcs.interactions.enumerate import (
    build_primitive_interaction_space,
    resolve_primitive_cutoff,
)
from mlfcs.interactions.realization import (
    InteractionAliasingError,
    validate_realization_identifiability,
)
from mlfcs.interactions.space import ReferenceFrame
from mlfcs.structure.supercell import build_supercell


def json_default(value):
    """Convert NumPy scalar diagnostics without weakening NaN rejection."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


@dataclass
class ClosureSystem:
    frame: ReferenceFrame
    space: object
    finite_basis: np.ndarray
    hessian_basis: np.ndarray
    observable_asr: np.ndarray
    observable_asr_basis: np.ndarray
    transferable_asr_basis: np.ndarray
    mapping: np.ndarray
    constrained_mapping: np.ndarray
    constrained_image: np.ndarray
    constrained_closure: np.ndarray

    @property
    def constrained_hessian_basis(self) -> np.ndarray:
        return self.hessian_basis @ self.observable_asr_basis

    @property
    def constrained_coordinate_map(self) -> np.ndarray:
        return self.observable_asr_basis @ np.column_stack(
            (self.constrained_mapping, self.constrained_closure)
        )


def build_system(primitive, size: int, cutoff: float) -> tuple[ClosureSystem, dict[str, object]]:
    started = time.perf_counter()
    reference = build_supercell(primitive, (size, size, size))
    frame = ReferenceFrame.from_atoms(primitive, reference, symprec=1e-5)
    finite_basis, _projector_rank = phase2.observable_basis(frame)
    projector_seconds = time.perf_counter() - started
    hessian_basis = phase2.full_hessian_basis(finite_basis, frame.relation)
    observable_asr = phase2.asr_constraint_matrix(hessian_basis, len(reference))
    observable_asr_basis, observable_asr_rank = phase2.null_space(observable_asr)
    space = build_primitive_interaction_space(
        primitive,
        order=2,
        cutoff=cutoff,
        max_body_order=None,
        symprec=1e-5,
        symmetry=frame.primitive_symmetry,
    )
    alias_error = None
    try:
        validate_realization_identifiability(space, frame.relation.index)
    except InteractionAliasingError as error:
        alias_error = str(error)
    mapping, projection_residual = phase2.transferable_map(
        space, frame.relation, finite_basis
    )
    transferable_constraints = build_translational_constraints(space).toarray()
    transferable_asr_basis, transferable_asr_rank = phase2.null_space(
        transferable_constraints
    )
    constrained_mapping = (
        observable_asr_basis.T @ mapping @ transferable_asr_basis
    )
    constrained_rank = phase2.numerical_rank(constrained_mapping)
    if constrained_rank.rank != constrained_mapping.shape[1]:
        raise RuntimeError("ASR-constrained transferable realization has a kernel")
    constrained_image, constrained_closure = phase2.closure_basis(
        constrained_mapping, constrained_rank.rank
    )
    system = ClosureSystem(
        frame,
        space,
        finite_basis,
        hessian_basis,
        observable_asr,
        observable_asr_basis,
        transferable_asr_basis,
        mapping,
        constrained_mapping,
        constrained_image,
        constrained_closure,
    )
    metrics = {
        "size": [size, size, size],
        "reference_atoms": len(reference),
        "raw_compact_dimension": int(finite_basis.shape[0]),
        "observable_dimension": int(finite_basis.shape[1]),
        "observable_asr_constraint_rank": observable_asr_rank.rank,
        "observable_asr_dimension": int(observable_asr_basis.shape[1]),
        "transferable_dimension": int(mapping.shape[1]),
        "transferable_asr_constraint_rank": transferable_asr_rank.rank,
        "transferable_asr_dimension": int(constrained_mapping.shape[1]),
        "transferable_asr_rank": constrained_rank.rank,
        "transferable_asr_kernel": int(
            constrained_mapping.shape[1] - constrained_rank.rank
        ),
        "closure_asr_dimension": int(constrained_closure.shape[1]),
        "transferable_projection_residual": projection_residual,
        "production_alias_check": "passed" if alias_error is None else "rejected",
        "production_alias_message": alias_error,
        "observable_projector_seconds": projector_seconds,
        "full_hessian_basis_mib": hessian_basis.nbytes / 2**20,
    }
    return system, metrics


def target_hessian(reference, calculator, step: float = 1e-4) -> np.ndarray:
    """Return the numerical Hessian from central force derivatives."""
    n_atoms = len(reference)
    hessian = np.empty((n_atoms, n_atoms, 3, 3), dtype=float)
    for atom in range(n_atoms):
        for direction in range(3):
            plus = reference.copy()
            minus = reference.copy()
            plus.positions[atom, direction] += step
            minus.positions[atom, direction] -= step
            plus.calc = calculator
            minus.calc = calculator
            derivative = (plus.get_forces() - minus.get_forces()) / (2.0 * step)
            hessian[:, atom, :, direction] = -derivative
    return hessian


def project_target(system: ClosureSystem, target: np.ndarray) -> dict[str, object]:
    basis = system.constrained_hessian_basis
    target_vector = target.reshape(-1)
    coordinates, *_ = np.linalg.lstsq(basis, target_vector, rcond=None)
    projected = basis @ coordinates
    transferable_coordinates = system.constrained_image @ (
        system.constrained_image.T @ coordinates
    )
    closure_coordinates = system.constrained_closure @ (
        system.constrained_closure.T @ coordinates
    )
    transferable = basis @ transferable_coordinates
    closure = basis @ closure_coordinates
    target_norm = float(np.linalg.norm(projected))
    n_atoms = len(system.frame.relation.reference)
    return {
        "target_full_frobenius_norm": float(np.linalg.norm(target_vector)),
        "target_allowed_frobenius_norm": target_norm,
        "projection_relative_residual": float(
            np.linalg.norm(target_vector - projected) / np.linalg.norm(target_vector)
        ),
        "transferable_frobenius_norm": float(np.linalg.norm(transferable)),
        "closure_frobenius_norm": float(np.linalg.norm(closure)),
        "closure_to_allowed_target_norm_ratio": float(
            np.linalg.norm(closure) / target_norm
        ),
        "allowed_norm_per_sqrt_atom": target_norm / np.sqrt(n_atoms),
        "closure_norm_per_sqrt_atom": float(np.linalg.norm(closure) / np.sqrt(n_atoms)),
        "decomposition_relative_residual": float(
            np.linalg.norm(projected - transferable - closure) / target_norm
        ),
        "allowed_coordinates": coordinates,
        "closure_full": closure,
    }


def pair_distance_distribution(system: ClosureSystem, closure_full: np.ndarray) -> dict[str, float]:
    relation = system.frame.relation
    full = closure_full.reshape(
        (len(relation.reference), len(relation.reference), 3, 3)
    )
    compact = compact_fc2(full, relation.reference)
    distribution: dict[str, float] = {}
    for site in range(len(relation.primitive)):
        anchor = relation.index.atom(site, (0, 0, 0))
        vectors = relation.reference.positions - relation.reference.positions[anchor]
        mic, distances = find_mic(vectors, relation.reference.cell, pbc=True)
        del mic
        for atom, distance in enumerate(distances):
            key = f"{float(distance):.6f}"
            value = float(np.linalg.norm(compact[site, atom]))
            distribution[key] = distribution.get(key, 0.0) + value * value
    return {key: float(np.sqrt(value)) for key, value in sorted(distribution.items())}


def decomposition_structure(system: ClosureSystem) -> dict[str, object]:
    _old_image, old_closure = phase2.closure_basis(
        system.mapping, phase2.numerical_rank(system.mapping).rank
    )
    action_t = system.observable_asr @ system.mapping
    action_c = system.observable_asr @ old_closure
    rank_t = phase2.numerical_rank(action_t).rank
    rank_c = phase2.numerical_rank(action_c).rank
    t_kernel, _ = phase2.null_space(action_t)
    c_kernel, _ = phase2.null_space(action_c)
    t_asr_ambient = system.mapping @ t_kernel
    c_asr_ambient = old_closure @ c_kernel
    constrained_closure_ambient = (
        system.observable_asr_basis @ system.constrained_closure
    )
    separate = np.column_stack((t_asr_ambient, c_asr_ambient))
    separate_rank = phase2.numerical_rank(separate).rank
    separate_intersection = (
        t_asr_ambient.shape[1] + c_asr_ambient.shape[1] - separate_rank
    )
    old_new_angles = subspace_angles(old_closure, constrained_closure_ambient)
    mixed_missing = (
        system.observable_asr_basis.shape[1] - separate_rank
    )
    return {
        "rank_A_restricted_to_transferable": rank_t,
        "rank_A_restricted_to_unconstrained_closure": rank_c,
        "dimension_kernel_A_intersect_transferable": int(t_kernel.shape[1]),
        "dimension_kernel_A_intersect_unconstrained_closure": int(c_kernel.shape[1]),
        "separately_constrained_sum_rank": separate_rank,
        "separately_constrained_intersection_dimension": separate_intersection,
        "asr_observable_dimension": int(system.observable_asr_basis.shape[1]),
        "mixed_cancellation_dimensions_missing_from_separate_constraints": mixed_missing,
        "old_vs_constrained_closure_principal_angles_radians": old_new_angles.tolist(),
        "old_vs_constrained_closure_intersection_dimension": int(
            old_closure.shape[1]
            + constrained_closure_ambient.shape[1]
            - phase2.numerical_rank(
                np.column_stack((old_closure, constrained_closure_ambient))
            ).rank
        ),
    }


def metric_and_commutation(system: ClosureSystem) -> dict[str, object]:
    full_gram = system.hessian_basis.T @ system.hessian_basis
    diagonal = np.diag(full_gram)
    off_diagonal = full_gram - np.diag(diagonal)
    raw_dimension = system.finite_basis.shape[0]
    raw_hessian_basis = phase2.full_hessian_basis(
        np.eye(raw_dimension), system.frame.relation
    )
    raw_asr = phase2.asr_constraint_matrix(
        raw_hessian_basis, len(system.frame.relation.reference)
    )
    raw_asr_null, _ = phase2.null_space(raw_asr)
    p_asr = raw_asr_null @ raw_asr_null.T
    p_sym = system.finite_basis @ system.finite_basis.T
    commutator = p_sym @ p_asr - p_asr @ p_sym
    full_metric_scale = float(np.mean(diagonal))
    mapping_ambient = system.observable_asr_basis @ system.constrained_mapping
    closure_ambient = system.observable_asr_basis @ system.constrained_closure
    return {
        "compact_basis_orthogonality_residual": float(
            np.linalg.norm(system.finite_basis.T @ system.finite_basis - np.eye(system.finite_basis.shape[1]))
        ),
        "full_hessian_gram_diagonal_min": float(np.min(diagonal)),
        "full_hessian_gram_diagonal_max": float(np.max(diagonal)),
        "full_hessian_gram_off_diagonal_norm": float(np.linalg.norm(off_diagonal)),
        "full_to_compact_metric_is_scalar_multiple": bool(
            np.max(diagonal) - np.min(diagonal) < 1e-10
            and np.linalg.norm(off_diagonal) < 1e-10
        ),
        "full_to_compact_metric_scale": full_metric_scale,
        "transferable_closure_compact_metric_orthogonality": float(
            np.linalg.norm(mapping_ambient.T @ closure_ambient, ord=2)
        ),
        "transferable_closure_full_frobenius_metric_orthogonality": float(
            np.linalg.norm(mapping_ambient.T @ full_gram @ closure_ambient, ord=2)
        ),
        "symmetry_asr_projector_commutator_2norm": float(
            np.linalg.norm(commutator, ord=2)
        ),
        "rank_symmetry_projector": phase2.numerical_rank(p_sym).rank,
        "rank_raw_asr_projector": phase2.numerical_rank(p_asr).rank,
        "rank_intersection": int(system.observable_asr_basis.shape[1]),
    }


def canonicality_study(system: ClosureSystem) -> dict[str, object]:
    """Separate the invariant closure subspace from arbitrary SVD coordinates."""
    closure = system.constrained_closure
    rng = np.random.default_rng(20260823)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(closure.shape[1], closure.shape[1])))
    rotated = closure @ orthogonal
    coefficients = rng.normal(size=closure.shape[1])
    original_hessian = system.constrained_hessian_basis @ closure @ coefficients
    rotated_hessian = (
        system.constrained_hessian_basis @ rotated @ orthogonal.T @ coefficients
    )
    return {
        "closure_dimension": int(closure.shape[1]),
        "subspace_projector_rotation_residual": float(
            np.linalg.norm(closure @ closure.T - rotated @ rotated.T)
        ),
        "coordinate_compensated_hessian_relative_residual": float(
            np.linalg.norm(original_hessian - rotated_hessian)
            / np.linalg.norm(original_hessian)
        ),
        "coordinate_statement": "eta is basis-dependent; the source Hessian and closure projector carry stable semantics",
        "svd_ambiguity": "the zero-singular-value complement is degenerate, so signs and internal rotations are backend-dependent",
    }


def fit_one(design: np.ndarray, forces: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    rank = phase2.numerical_rank(design)
    parameters, *_ = np.linalg.lstsq(design, forces.reshape(-1), rcond=None)
    prediction = design @ parameters
    singular = rank.singular_values
    covariance = np.linalg.inv(design.T @ design)
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    return parameters, {
        "rank": rank.rank,
        "columns": design.shape[1],
        "smallest_singular_value": float(singular[-1]),
        "condition_number": float(singular[0] / singular[-1]),
        "force_rmse_eV_per_angstrom": float(
            np.sqrt(np.mean(np.square(prediction - forces.reshape(-1))))
        ),
        "maximum_covariance_proxy_diagonal": float(np.max(np.diag(covariance))),
        "maximum_absolute_parameter_correlation": float(
            np.max(np.abs(correlation - np.eye(len(correlation))))
        ),
    }


def robustness_sweep(system: ClosureSystem, reference, calculator) -> dict[str, object]:
    rows = []
    coordinate_map = system.constrained_coordinate_map
    transfer_count = system.constrained_mapping.shape[1]
    for seed in (7, 42, 2026):
        for sigma in (0.003, 0.01, 0.03):
            rng = np.random.default_rng(seed)
            displacements = rng.normal(scale=sigma, size=(100, len(reference), 3))
            displacements -= displacements.mean(axis=1, keepdims=True)
            forces = phase2.evaluate_forces(reference, displacements, calculator)
            full_design = phase2.design_from_hessian_basis(
                displacements, system.hessian_basis
            )
            design = full_design @ coordinate_map
            for frames in (10, 25, 50, 100):
                row_design = design[: frames * len(reference) * 3]
                row_forces = forces[:frames]
                parameters, metrics = fit_one(row_design, row_forces)
                t_norm = float(np.linalg.norm(parameters[:transfer_count]))
                c_norm = float(np.linalg.norm(parameters[transfer_count:]))
                metrics.update(
                    {
                        "seed": seed,
                        "sigma_angstrom": sigma,
                        "frames": frames,
                        "transferable_parameter_norm": t_norm,
                        "closure_parameter_norm": c_norm,
                    }
                )
                rows.append(metrics)
    return {"rows": rows}


def noise_study(system: ClosureSystem, reference, calculator) -> dict[str, object]:
    rng = np.random.default_rng(42)
    displacements = rng.normal(scale=0.01, size=(100, len(reference), 3))
    displacements -= displacements.mean(axis=1, keepdims=True)
    forces = phase2.evaluate_forces(reference, displacements, calculator)
    design = (
        phase2.design_from_hessian_basis(displacements, system.hessian_basis)
        @ system.constrained_coordinate_map
    )
    baseline, base_metrics = fit_one(design, forces)
    rows = []
    for noise in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        noise_rng = np.random.default_rng(20260823)
        perturbed = forces + noise_rng.normal(scale=noise, size=forces.shape)
        parameters, metrics = fit_one(design, perturbed)
        rows.append(
            {
                "force_noise_sigma_eV_per_angstrom": noise,
                "relative_parameter_change": float(
                    np.linalg.norm(parameters - baseline) / np.linalg.norm(baseline)
                ),
                "absolute_parameter_change": float(np.linalg.norm(parameters - baseline)),
                "fit_rmse_eV_per_angstrom": metrics["force_rmse_eV_per_angstrom"],
            }
        )
    transfer_count = system.constrained_mapping.shape[1]
    normalized = design / np.linalg.norm(design, axis=0)
    cross = normalized[:, :transfer_count].T @ normalized[:, transfer_count:]
    return {
        "baseline": base_metrics,
        "transferable_closure_normalized_cross_block_2norm": float(
            np.linalg.norm(cross, ord=2)
        ),
        "rows": rows,
    }


def cutoff_sweep(primitive, base_system: ClosureSystem, target, displacements, forces):
    finite_basis = base_system.finite_basis
    asr_basis = base_system.observable_asr_basis
    hessian_basis = base_system.hessian_basis
    finite_design = phase2.design_from_hessian_basis(displacements, hessian_basis)
    target_vector = target.reshape(-1)
    target_coordinates, *_ = np.linalg.lstsq(
        base_system.constrained_hessian_basis, target_vector, rcond=None
    )
    rows = []
    for cutoff in (2.5, 3.2, 3.8, 4.2, 4.439115867225757, 4.8, 5.5, 6.0):
        space = build_primitive_interaction_space(
            primitive,
            order=2,
            cutoff=cutoff,
            max_body_order=None,
            symprec=1e-5,
            symmetry=base_system.frame.primitive_symmetry,
        )
        alias = False
        try:
            validate_realization_identifiability(space, base_system.frame.relation.index)
        except InteractionAliasingError:
            alias = True
        mapping, _ = phase2.transferable_map(
            space, base_system.frame.relation, finite_basis
        )
        constraints = build_translational_constraints(space).toarray()
        z_theta, constraint_rank = phase2.null_space(constraints)
        constrained_mapping = asr_basis.T @ mapping @ z_theta
        map_rank = phase2.numerical_rank(constrained_mapping)
        kernel = constrained_mapping.shape[1] - map_rank.rank
        row = {
            "cutoff_angstrom": cutoff,
            "transferable_dimension": int(mapping.shape[1]),
            "transferable_asr_dimension": int(constrained_mapping.shape[1]),
            "transferable_constraint_rank": constraint_rank.rank,
            "transferable_asr_rank": map_rank.rank,
            "transferable_asr_kernel": kernel,
            "production_alias_check_rejected": alias,
        }
        if kernel:
            row["status"] = "rejected: constrained transferable alias"
            rows.append(row)
            continue
        image, closure = phase2.closure_basis(constrained_mapping, map_rank.rank)
        target_transferable = image @ (image.T @ target_coordinates)
        target_closure = closure @ (closure.T @ target_coordinates)
        coordinate_map = asr_basis @ np.column_stack((constrained_mapping, closure))
        design = finite_design @ coordinate_map
        parameters, fit = fit_one(design, forces)
        del parameters
        row.update(
            {
                "status": "usable",
                "closure_dimension": int(closure.shape[1]),
                "target_transferable_hessian_norm": float(
                    np.linalg.norm(base_system.constrained_hessian_basis @ target_transferable)
                ),
                "target_closure_hessian_norm": float(
                    np.linalg.norm(base_system.constrained_hessian_basis @ target_closure)
                ),
                "target_closure_norm_ratio": float(
                    np.linalg.norm(target_closure) / np.linalg.norm(target_coordinates)
                ),
                "joint_design_rank": fit["rank"],
                "joint_design_columns": fit["columns"],
                "joint_design_condition_number": fit["condition_number"],
                "force_rmse_eV_per_angstrom": fit["force_rmse_eV_per_angstrom"],
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

    primitive = ase_from_phonopy(load(HARMONIC_PATH).primitive)
    reference2 = build_supercell(primitive, (2, 2, 2))
    cutoff = resolve_primitive_cutoff(primitive, None, reference=reference2)
    calculator = PolymlpASECalculator(pot=POTENTIAL_PATH)

    system2, reference2_metrics = build_system(primitive, 2, cutoff)
    target2 = target_hessian(reference2, calculator)
    target2_analysis = project_target(system2, target2)
    closure_distribution = pair_distance_distribution(
        system2, target2_analysis.pop("closure_full")
    )
    target2_analysis.pop("allowed_coordinates")

    rng = np.random.default_rng(42)
    displacements = rng.normal(scale=0.01, size=(100, len(reference2), 3))
    displacements -= displacements.mean(axis=1, keepdims=True)
    forces = phase2.evaluate_forces(reference2, displacements, calculator)

    system3, reference3_metrics = build_system(primitive, 3, cutoff)
    reference3 = system3.frame.relation.reference
    target3 = target_hessian(reference3, calculator)
    target3_analysis = project_target(system3, target3)
    target3_analysis.pop("closure_full")
    target3_analysis.pop("allowed_coordinates")

    transferable_constraints = build_translational_constraints(system2.space)
    production_parameter_map = explicit_constraint_null_space(
        transferable_constraints, tolerance=1e-11
    ).toarray()
    simple_parameter_map = np.asarray(
        [[-4.0, -2.0], [1.0, 0.0], [0.0, 1.0], [-4.0, -2.0]]
    )
    probe_theta = production_parameter_map @ np.asarray([0.37, -0.19])
    probe_sparse = expand_primitive_parameters(system2.space, probe_theta)
    key_digest = sha256()
    key_digest.update(probe_sparse.sites.tobytes())
    key_digest.update(probe_sparse.translations.tobytes())

    results = {
        "scope": "research prototype only; no production implementation",
        "base_cutoff_angstrom": cutoff,
        "dimension_structure": decomposition_structure(system2),
        "metric_and_operator_compatibility": metric_and_commutation(system2),
        "closure_canonicality": canonicality_study(system2),
        "transferable_asr_map": {
            "constraint_matrix": transferable_constraints.toarray().tolist(),
            "reduced_coordinate_map_theta_equals_Rz": system2.transferable_asr_basis.tolist(),
            "production_qr_parameter_map": production_parameter_map.tolist(),
            "production_vs_svd_max_principal_angle_radians": float(
                np.max(subspace_angles(production_parameter_map, system2.transferable_asr_basis))
            ),
            "production_constraint_residual": float(
                np.linalg.norm(transferable_constraints @ production_parameter_map)
            ),
            "simple_equivalent_map_using_z_equals_theta_1_theta_2": simple_parameter_map.tolist(),
            "simple_map_constraint_residual": float(
                np.linalg.norm(transferable_constraints @ simple_parameter_map)
            ),
            "simple_vs_production_max_principal_angle_radians": float(
                np.max(subspace_angles(simple_parameter_map, production_parameter_map))
            ),
            "independent_constraint_equations": [
                "theta_0 + 4 theta_1 + 2 theta_2 = 0",
                "4 theta_1 + 2 theta_2 + theta_3 = 0",
            ],
            "orbit_descriptions": [
                {
                    "representative_labels": [list(label) for label in orbit.representative.labels],
                    "dimension": orbit.dimension,
                    "pivots": list(orbit.pivots),
                    "invariant_basis_columns": orbit.basis.tolist(),
                }
                for orbit in system2.space.orbits
            ],
            "exact_r_reconstruction": {
                "sparse_rows": len(probe_sparse.sites),
                "site_translation_key_sha256": key_digest.hexdigest(),
                "constraint_residual": float(
                    np.linalg.norm(transferable_constraints @ probe_theta)
                ),
                "semantics": "ASR changes only reduced coordinates; exact-R sites and translations remain those of the current transferable space",
            },
        },
        "reference_sweep": {
            "rows": [reference2_metrics, reference3_metrics],
            "four_by_four": {
                "status": "not materialized",
                "reason": "the current dense research projector scales poorly and is not a production requirement",
                "reference_atoms": 128,
                "raw_compact_dimension": 2304,
                "dense_projector_mib": 2304 * 2304 * 8 / 2**20,
            },
            "target_hessian": {
                "2x2x2": target2_analysis,
                "3x3x3": target3_analysis,
            },
            "canonical_injection": {
                "exists": False,
                "reason": "a finite residue class has multiple exact-R lifts; choosing one requires extra real-space semantics not contained in the closure",
            },
        },
        "cutoff_sweep": cutoff_sweep(
            primitive, system2, target2, displacements, forces
        ),
        "target_hessian_pair_distance_distribution": closure_distribution,
        "dataset_robustness": robustness_sweep(
            system2, reference2, calculator
        ),
        "conditioning_and_noise": noise_study(system2, reference2, calculator),
        "degenerate_conditions": {
            "primitive_alias": "reject with InteractionAliasingError",
            "constrained_transferable_alias": "reject",
            "zero_closure_dimension": "reduce exactly to current transferable FC2",
            "joint_dataset_rank_deficient": "reject without regularization",
            "source_reference_mismatch": "closure is not interpretable or exportable",
        },
    }
    RESULTS.write_text(
        json.dumps(results, indent=2, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "reference_sweep": results["reference_sweep"],
        "dimension_structure": results["dimension_structure"],
        "metric_and_operator_compatibility": results["metric_and_operator_compatibility"],
        "results": str(RESULTS),
    }, indent=2, allow_nan=False, default=json_default))


if __name__ == "__main__":
    main()
