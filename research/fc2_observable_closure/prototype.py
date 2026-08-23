#!/usr/bin/env python3
"""Minimal FC2 transferable-image and finite-supercell-closure experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from phonopy import load
from scipy.linalg import subspace_angles

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "examples" / "sscha" / "KCl"
RESULTS = Path(__file__).with_name("results.json")
if str(CASE) not in sys.path:
    sys.path.insert(0, str(CASE))

from common import HARMONIC_PATH, POTENTIAL_PATH, ase_from_phonopy

from mlfcs.constraints.translational import build_translational_constraints
from mlfcs.force_constants.dense import compact_fc2, expand_compact_fc2
from mlfcs.force_constants.expansion import expand_primitive_parameters
from mlfcs.interactions.enumerate import build_primitive_interaction_space
from mlfcs.interactions.realization import (
    InteractionAliasingError,
    validate_realization_identifiability,
)
from mlfcs.interactions.space import ReferenceFrame
from mlfcs.structure.relation import StructureRelation
from mlfcs.structure.supercell import build_supercell


@dataclass(frozen=True)
class RankResult:
    rank: int
    tolerance: float
    singular_values: np.ndarray


def numerical_rank(values: np.ndarray) -> RankResult:
    singular = np.linalg.svd(np.asarray(values, dtype=float), compute_uv=False)
    largest = float(singular[0]) if len(singular) else 0.0
    tolerance = np.finfo(float).eps * max(values.shape, default=0) * largest
    return RankResult(int(np.count_nonzero(singular > tolerance)), tolerance, singular)


def null_space(values: np.ndarray) -> tuple[np.ndarray, RankResult]:
    """Return an orthonormal null-space basis using the shared rank convention."""
    matrix = np.asarray(values, dtype=float)
    _left, singular, right = np.linalg.svd(matrix, full_matrices=True)
    rank = numerical_rank(matrix)
    return right[rank.rank :].T, RankResult(rank.rank, rank.tolerance, singular)


def rank_stability(values: np.ndarray) -> dict[str, int]:
    singular = np.linalg.svd(np.asarray(values, dtype=float), compute_uv=False)
    largest = float(singular[0]) if len(singular) else 0.0
    base = np.finfo(float).eps * max(values.shape, default=0) * largest
    return {
        str(multiplier): int(np.count_nonzero(singular > multiplier * base))
        for multiplier in (0.1, 1.0, 10.0, 100.0)
    }


def compact_from_sparse(space, parameters, relation) -> np.ndarray:
    sparse = expand_primitive_parameters(space, parameters)
    compact = np.zeros((len(relation.primitive), len(relation.reference), 3, 3))
    for sites, translations, tensor in zip(
        sparse.sites, sparse.translations, sparse.tensors, strict=True
    ):
        atom = relation.index.atom(int(sites[1]), translations[0])
        compact[int(sites[0]), atom] += tensor
    return compact


def symmetrize_full_fc2(full: np.ndarray, frame: ReferenceFrame) -> np.ndarray:
    """Project one full Hessian onto permutation and finite space-group invariance."""
    values = 0.5 * (full + full.transpose(1, 0, 3, 2))
    projected = np.zeros_like(values)
    for permutation, rotation in zip(
        frame.symmetry.atom_permutations,
        frame.symmetry.cartesian_rotations,
        strict=True,
    ):
        # Primitive orbit TensorAction uses the passive Cartesian action R.T.
        # The finite Hessian projector must use the same convention.
        rotation = rotation.T
        transformed = np.einsum(
            "ac,ijcd,bd->ijab", rotation, values, rotation, optimize=True
        )
        image = np.empty_like(values)
        image[np.ix_(permutation, permutation)] = transformed
        projected += image
    return projected / frame.symmetry.size


def observable_basis(frame: ReferenceFrame) -> tuple[np.ndarray, RankResult]:
    """Return an orthonormal basis of symmetry-allowed compact finite FC2."""
    relation = frame.relation
    shape = (len(relation.primitive), len(relation.reference), 3, 3)
    dimension = int(np.prod(shape))
    projector = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        compact = np.zeros(shape, dtype=float)
        compact.reshape(-1)[column] = 1.0
        full = expand_compact_fc2(compact, relation.reference)
        projected = symmetrize_full_fc2(full, frame)
        projector[:, column] = compact_fc2(projected, relation.reference).reshape(-1)
    left, singular, _right = np.linalg.svd(projector, full_matrices=False)
    rank = numerical_rank(projector)
    basis = left[:, : rank.rank]
    # The averaged group action should be an orthogonal projector.  Keep this
    # check explicit because an atom-permutation or Cartesian convention error
    # would otherwise create a plausible but incorrect observable dimension.
    idempotence = float(np.linalg.norm(projector @ projector - projector))
    symmetry = float(np.linalg.norm(projector - projector.T))
    if idempotence > 1e-9 or symmetry > 1e-9:
        raise RuntimeError(
            "finite Hessian group average is not an orthogonal projector: "
            f"idempotence={idempotence:.3e}, symmetry={symmetry:.3e}"
        )
    return basis, RankResult(rank.rank, rank.tolerance, singular)


def transferable_map(space, relation, basis) -> tuple[np.ndarray, float]:
    n_parameters = sum(orbit.dimension for orbit in space.orbits)
    values = np.empty((basis.shape[0], n_parameters), dtype=float)
    residual = 0.0
    for column in range(n_parameters):
        parameters = np.zeros(n_parameters)
        parameters[column] = 1.0
        compact = compact_from_sparse(space, parameters, relation).reshape(-1)
        coordinates = basis.T @ compact
        values[:, column] = basis @ coordinates
        residual = max(residual, float(np.linalg.norm(compact - values[:, column])))
    return basis.T @ values, residual


def closure_basis(mapping: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    left, _singular, _right = np.linalg.svd(mapping, full_matrices=True)
    return left[:, :rank], left[:, rank:]


def full_hessian_basis(observable: np.ndarray, relation) -> np.ndarray:
    full = []
    shape = (len(relation.primitive), len(relation.reference), 3, 3)
    for column in observable.T:
        compact = column.reshape(shape)
        full.append(expand_compact_fc2(compact, relation.reference).reshape(-1))
    return np.asarray(full).T


def asr_constraint_matrix(hessian_basis: np.ndarray, n_atoms: int) -> np.ndarray:
    """Map Hessian-basis coordinates to sum_j Phi[i,a,j,b]."""
    hessians = hessian_basis.T.reshape((-1, n_atoms, n_atoms, 3, 3))
    return np.sum(hessians, axis=2).reshape((len(hessians), -1)).T


def design_from_hessian_basis(displacements: np.ndarray, hessian_basis: np.ndarray) -> np.ndarray:
    n_atoms = displacements.shape[1]
    hessians = hessian_basis.T.reshape((-1, n_atoms, n_atoms, 3, 3))
    forces = -np.einsum("pijab,sjb->siap", hessians, displacements, optimize=True)
    return forces.reshape((-1, len(hessians)))


def evaluate_forces(reference: Atoms, displacements: np.ndarray, calculator) -> np.ndarray:
    forces = np.empty_like(displacements)
    for snapshot, displacement in enumerate(displacements):
        atoms = reference.copy()
        atoms.positions += displacement
        atoms.calc = calculator
        forces[snapshot] = atoms.get_forces()
    return forces


def kcl_structures(
    snapshots: int,
) -> tuple[Atoms, Atoms, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phonon = load(HARMONIC_PATH)
    primitive = ase_from_phonopy(phonon.primitive)
    reference = build_supercell(primitive, (2, 2, 2))
    rng = np.random.default_rng(42)
    uncentered = rng.normal(scale=0.01, size=(snapshots, len(reference), 3))
    centered = uncentered - uncentered.mean(axis=1, keepdims=True)
    from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

    calculator = PolymlpASECalculator(pot=POTENTIAL_PATH)
    centered_forces = evaluate_forces(reference, centered, calculator)
    uncentered_forces = evaluate_forces(reference, uncentered, calculator)
    return primitive, reference, centered, centered_forces, uncentered, uncentered_forces


def aliasing_negative_control() -> dict[str, object]:
    primitive = Atoms("Si", scaled_positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
    reference = primitive.copy()
    relation = StructureRelation.from_atoms(primitive, reference)
    space = build_primitive_interaction_space(
        primitive,
        order=2,
        cutoff=4.1,
        max_body_order=None,
        symprec=1e-5,
    )
    dimension = sum(orbit.dimension for orbit in space.orbits)
    columns = []
    for parameter in range(dimension):
        values = np.zeros(dimension)
        values[parameter] = 1.0
        columns.append(compact_from_sparse(space, values, relation).reshape(-1))
    mapping = np.asarray(columns).T
    rank = numerical_rank(mapping)
    rejected = False
    try:
        validate_realization_identifiability(space, relation.index)
    except InteractionAliasingError:
        rejected = True
    return {
        "primitive_parameter_dimension": dimension,
        "realization_rank": rank.rank,
        "kernel_dimension": dimension - rank.rank,
        "production_check_rejected": rejected,
        "singular_values": rank.singular_values.tolist(),
    }


def run(snapshots: int) -> dict[str, object]:
    (
        primitive,
        reference,
        centered_displacements,
        centered_forces,
        uncentered_displacements,
        _uncentered_forces,
    ) = kcl_structures(snapshots)
    frame = ReferenceFrame.from_atoms(primitive, reference, symprec=1e-5)
    # Resolve cutoff=None through the same public interaction-space semantics.
    from mlfcs.interactions.enumerate import resolve_primitive_cutoff

    cutoff = resolve_primitive_cutoff(primitive, None, reference=reference)
    space = build_primitive_interaction_space(
        primitive,
        order=2,
        cutoff=cutoff,
        max_body_order=None,
        symprec=1e-5,
        symmetry=frame.primitive_symmetry,
    )
    validate_realization_identifiability(space, frame.relation.index)

    finite_basis, projector_rank = observable_basis(frame)
    mapping, map_residual = transferable_map(space, frame.relation, finite_basis)
    map_rank = numerical_rank(mapping)
    parameter_dimension = mapping.shape[1]
    kernel_dimension = parameter_dimension - map_rank.rank
    closure_dimension = finite_basis.shape[1] - map_rank.rank

    result: dict[str, object] = {
        "case": "KCl primitive 2 atoms, 2x2x2 reference",
        "primitive_atoms": len(primitive),
        "reference_atoms": len(reference),
        "supercell_matrix": frame.relation.supercell_matrix.tolist(),
        "cutoff_angstrom": cutoff,
        "snapshots": snapshots,
        "primitive_parameter_dimension": parameter_dimension,
        "realization_rank": map_rank.rank,
        "kernel_dimension": kernel_dimension,
        "observable_dimension": finite_basis.shape[1],
        "closure_dimension": closure_dimension,
        "mapping_singular_values": map_rank.singular_values.tolist(),
        "mapping_rank_tolerance": map_rank.tolerance,
        "transferable_projection_residual": map_residual,
        "observable_projector_rank_tolerance": projector_rank.tolerance,
        "mapping_rank_stability": rank_stability(mapping),
    }
    if kernel_dimension != 0 or closure_dimension <= 0:
        result["decision"] = "NO-GO"
        result["reason"] = (
            "transferable map has a kernel"
            if kernel_dimension
            else "transferable image already spans the finite observable space"
        )
        result["negative_control"] = aliasing_negative_control()
        return result

    _image, closure = closure_basis(mapping, map_rank.rank)
    combined = np.column_stack((mapping, closure))
    combined_rank = numerical_rank(combined)
    hessian_basis = full_hessian_basis(finite_basis, frame.relation)

    # Define the physical observable space first, then take null(C_ASR).
    observable_asr = asr_constraint_matrix(hessian_basis, len(reference))
    observable_asr_basis, observable_asr_rank = null_space(observable_asr)
    constrained_observable_dimension = observable_asr_basis.shape[1]
    constrained_finite_basis = finite_basis @ observable_asr_basis
    constrained_hessian_basis = hessian_basis @ observable_asr_basis

    # Preserve the current transferable-ASR semantics by using the production
    # primitive constraint matrix, then verify it agrees with finite-Hessian ASR.
    transferable_asr = build_translational_constraints(space).toarray()
    transferable_asr_basis, transferable_asr_rank = null_space(transferable_asr)
    direct_transferable_asr_basis, _direct_transferable_asr_rank = null_space(
        observable_asr @ mapping
    )
    asr_basis_angles = subspace_angles(
        transferable_asr_basis, direct_transferable_asr_basis
    )
    asr_mapping_leakage = float(
        np.linalg.norm(observable_asr @ mapping @ transferable_asr_basis)
    )

    constrained_mapping = (
        observable_asr_basis.T @ mapping @ transferable_asr_basis
    )
    constrained_map_rank = numerical_rank(constrained_mapping)
    constrained_kernel = constrained_mapping.shape[1] - constrained_map_rank.rank
    constrained_closure_dimension = (
        constrained_observable_dimension - constrained_map_rank.rank
    )
    if constrained_kernel:
        result.update(
            {
                "decision": "NO-GO",
                "reason": "ASR-constrained transferable map has a kernel",
                "asr": {
                    "observable_constraint_rank": observable_asr_rank.rank,
                    "observable_dimension": constrained_observable_dimension,
                    "transferable_dimension": constrained_mapping.shape[1],
                    "transferable_rank": constrained_map_rank.rank,
                    "transferable_kernel_dimension": constrained_kernel,
                },
                "negative_control": aliasing_negative_control(),
            }
        )
        return result

    constrained_image, constrained_closure = closure_basis(
        constrained_mapping, constrained_map_rank.rank
    )
    constrained_combined = np.column_stack(
        (constrained_mapping, constrained_closure)
    )
    constrained_combined_rank = numerical_rank(constrained_combined)

    # Flow A is diagnostic only: project the old closure after construction.
    projected_old_closure = observable_asr_basis.T @ closure
    projected_old_closure_rank = numerical_rank(projected_old_closure)
    flow_a_combined = np.column_stack((constrained_mapping, projected_old_closure))
    flow_a_rank = numerical_rank(flow_a_combined)
    flow_a_intersection = (
        constrained_mapping.shape[1]
        + projected_old_closure_rank.rank
        - flow_a_rank.rank
    )
    flow_a_angles = subspace_angles(projected_old_closure, constrained_closure)

    rng = np.random.default_rng(20260823)
    constrained_target = rng.normal(size=constrained_observable_dimension)
    constrained_recovered, *_ = np.linalg.lstsq(
        constrained_combined, constrained_target, rcond=None
    )
    constrained_reconstruction = constrained_combined @ constrained_recovered
    constrained_coordinate_error = float(
        np.linalg.norm(constrained_reconstruction - constrained_target)
        / np.linalg.norm(constrained_target)
    )
    target_hessian = constrained_hessian_basis @ constrained_target
    recovered_hessian = constrained_hessian_basis @ constrained_reconstruction
    constrained_hessian_error = float(
        np.linalg.norm(recovered_hessian - target_hessian)
        / np.linalg.norm(target_hessian)
    )
    target_hessian_array = target_hessian.reshape(
        (len(reference), len(reference), 3, 3)
    )
    random_asr_residual = float(
        np.max(np.abs(np.sum(target_hessian_array, axis=1)))
    )
    random_permutation_residual = float(
        np.linalg.norm(target_hessian_array - target_hessian_array.transpose(1, 0, 3, 2))
    )
    random_symmetry_residual = float(
        np.linalg.norm(symmetrize_full_fc2(target_hessian_array, frame) - target_hessian_array)
    )

    centered_finite_design = design_from_hessian_basis(
        centered_displacements, hessian_basis
    )
    uncentered_finite_design = design_from_hessian_basis(
        uncentered_displacements, hessian_basis
    )
    centered_unconstrained = centered_finite_design @ combined
    uncentered_unconstrained = uncentered_finite_design @ combined
    constrained_coordinate_map = observable_asr_basis @ constrained_combined
    centered_constrained = centered_finite_design @ constrained_coordinate_map
    uncentered_constrained = uncentered_finite_design @ constrained_coordinate_map

    centered_transferable = centered_finite_design @ mapping
    centered_constrained_transferable = (
        centered_finite_design
        @ observable_asr_basis
        @ constrained_mapping
    )
    centered_constrained_closure = (
        centered_finite_design
        @ observable_asr_basis
        @ constrained_closure
    )

    control_designs = {
        "com_removed_unconstrained": centered_unconstrained,
        "com_retained_unconstrained": uncentered_unconstrained,
        "com_removed_asr_constrained": centered_constrained,
        "com_retained_asr_constrained": uncentered_constrained,
    }
    control = {}
    for label, design in control_designs.items():
        rank = numerical_rank(design)
        condition = (
            float(rank.singular_values[0] / rank.singular_values[rank.rank - 1])
            if rank.rank
            else float("inf")
        )
        control[label] = {
            "columns": design.shape[1],
            "rank": rank.rank,
            "nullity": design.shape[1] - rank.rank,
            "rank_tolerance": rank.tolerance,
            "condition_number_nonzero_subspace": condition,
            "singular_values": rank.singular_values.tolist(),
            "rank_stability": rank_stability(design),
        }

    # Reconstruct the previous phase's deterministic null vector and test its
    # projection into the newly defined ASR-constrained observable space.
    _left, previous_singular, previous_right = np.linalg.svd(
        centered_unconstrained, full_matrices=False
    )
    previous_rank = numerical_rank(centered_unconstrained)
    previous_null_parameters = previous_right[previous_rank.rank :].T[:, 0]
    previous_null_coordinates = combined @ previous_null_parameters
    previous_null_norm = float(np.linalg.norm(previous_null_coordinates))
    previous_null_projection = (
        observable_asr_basis @ observable_asr_basis.T @ previous_null_coordinates
    )
    previous_projection_ratio = float(
        np.linalg.norm(previous_null_projection) / previous_null_norm
    )
    previous_null_hessian = (hessian_basis @ previous_null_coordinates).reshape(
        (len(reference), len(reference), 3, 3)
    )
    uniform_displacements = np.zeros((3, len(reference), 3))
    for direction in range(3):
        uniform_displacements[direction, :, direction] = 1.0
    uniform_response = -np.einsum(
        "ijab,sjb->sia", previous_null_hessian, uniform_displacements, optimize=True
    )

    def fit_metrics(design, forces, coordinate_map):
        rank = numerical_rank(design)
        parameters, *_ = np.linalg.lstsq(design, forces.reshape(-1), rcond=None)
        prediction = design @ parameters
        coordinates = coordinate_map @ parameters
        hessian = (hessian_basis @ coordinates).reshape(
            (len(reference), len(reference), 3, 3)
        )
        return {
            "columns": design.shape[1],
            "rank": rank.rank,
            "nullity": design.shape[1] - rank.rank,
            "force_rmse_eV_per_angstrom": float(
                np.sqrt(np.mean(np.square(prediction - forces.reshape(-1))))
            ),
            "hessian_asr_maximum": float(np.max(np.abs(np.sum(hessian, axis=1)))),
            "minimum_norm_is_unique": rank.rank == design.shape[1],
            "parameters": parameters,
            "observable_coordinates": coordinates,
        }

    fit_a = fit_metrics(centered_transferable, centered_forces, mapping)
    fit_b = fit_metrics(centered_unconstrained, centered_forces, combined)
    fit_c = fit_metrics(
        centered_constrained,
        centered_forces,
        constrained_coordinate_map,
    )
    constrained_parameters = fit_c.pop("parameters")
    fit_c.pop("observable_coordinates")
    transferable_count = constrained_mapping.shape[1]
    fitted_transferable_coordinates = observable_asr_basis @ (
        constrained_mapping @ constrained_parameters[:transferable_count]
    )
    fitted_closure_coordinates = observable_asr_basis @ (
        constrained_closure @ constrained_parameters[transferable_count:]
    )
    fitted_total_hessian = hessian_basis @ (
        fitted_transferable_coordinates + fitted_closure_coordinates
    )
    fitted_closure_hessian = hessian_basis @ fitted_closure_coordinates
    closure_norm_ratio = float(
        np.linalg.norm(fitted_closure_hessian) / np.linalg.norm(fitted_total_hessian)
    )
    fit_a.pop("parameters")
    fit_a.pop("observable_coordinates")
    fit_b.pop("parameters")
    fit_b.pop("observable_coordinates")

    constrained_transferable_rank = numerical_rank(centered_constrained_transferable)
    constrained_closure_rank = numerical_rank(centered_constrained_closure)
    constrained_joint_rank = numerical_rank(centered_constrained)
    constrained_dataset_angles = subspace_angles(
        centered_constrained_transferable, centered_constrained_closure
    )
    observable_asr_rank_stability = rank_stability(observable_asr)
    transferable_rank_stability = rank_stability(constrained_mapping)
    representation_rank_stability = rank_stability(constrained_combined)
    dataset_rank_stability = rank_stability(centered_constrained)
    all_go = (
        np.linalg.norm(observable_asr @ observable_asr_basis) < 1e-12
        and np.linalg.norm(
            constrained_finite_basis.T @ constrained_finite_basis
            - np.eye(constrained_observable_dimension)
        )
        < 1e-12
        and constrained_kernel == 0
        and constrained_combined_rank.rank == constrained_observable_dimension
        and flow_a_intersection == 0
        and constrained_joint_rank.rank == centered_constrained.shape[1]
        and previous_projection_ratio < 1e-12
        and random_asr_residual < 1e-12
        and random_permutation_residual < 1e-12
        and random_symmetry_residual < 1e-12
        and fit_c["minimum_norm_is_unique"]
        and fit_c["hessian_asr_maximum"] < 1e-12
        and len(set(observable_asr_rank_stability.values())) == 1
        and len(set(transferable_rank_stability.values())) == 1
        and len(set(representation_rank_stability.values())) == 1
        and len(set(dataset_rank_stability.values())) == 1
    )
    result.update(
        {
            "unconstrained": {
                "combined_rank": combined_rank.rank,
                "combined_columns": combined.shape[1],
                "mapping_closure_orthogonality_2norm": float(
                    np.linalg.norm(mapping.T @ closure, ord=2)
                ),
            },
            "asr": {
                "observable_constraint_rows": observable_asr.shape[0],
                "observable_constraint_rank": observable_asr_rank.rank,
                "observable_constraint_singular_values": observable_asr_rank.singular_values.tolist(),
                "observable_constraint_rank_stability": observable_asr_rank_stability,
                "observable_dimension_before": finite_basis.shape[1],
                "observable_dimension_after": constrained_observable_dimension,
                "observable_basis_orthogonality": float(
                    np.linalg.norm(
                        constrained_finite_basis.T @ constrained_finite_basis
                        - np.eye(constrained_observable_dimension)
                    )
                ),
                "observable_basis_asr_residual": float(
                    np.linalg.norm(observable_asr @ observable_asr_basis)
                ),
                "observable_null_space_basis": observable_asr_basis.tolist(),
                "transferable_parameter_dimension_before": mapping.shape[1],
                "transferable_constraint_rank": transferable_asr_rank.rank,
                "transferable_parameter_dimension_after": constrained_mapping.shape[1],
                "transferable_map_rank": constrained_map_rank.rank,
                "transferable_kernel_dimension": constrained_kernel,
                "transferable_smallest_retained_singular_value": float(
                    constrained_map_rank.singular_values[constrained_map_rank.rank - 1]
                ),
                "transferable_rank_tolerance": constrained_map_rank.tolerance,
                "transferable_rank_stability": transferable_rank_stability,
                "production_vs_direct_asr_max_principal_angle": float(
                    np.max(asr_basis_angles) if len(asr_basis_angles) else 0.0
                ),
                "transferable_asr_mapping_residual": asr_mapping_leakage,
                "transferable_null_space_basis": transferable_asr_basis.tolist(),
                "closure_dimension": constrained_closure_dimension,
                "combined_rank": constrained_combined_rank.rank,
                "combined_columns": constrained_combined.shape[1],
                "mapping_closure_orthogonality_2norm": float(
                    np.linalg.norm(constrained_mapping.T @ constrained_closure, ord=2)
                ),
                "minimum_representation_principal_angle_radians": float(
                    np.min(subspace_angles(constrained_image, constrained_closure))
                ),
                "random_coordinate_reconstruction_relative_error": constrained_coordinate_error,
                "random_hessian_reconstruction_relative_error": constrained_hessian_error,
                "random_hessian_asr_maximum": random_asr_residual,
                "random_hessian_permutation_residual": random_permutation_residual,
                "random_hessian_symmetry_residual": random_symmetry_residual,
                "combined_rank_stability": representation_rank_stability,
            },
            "flow_a_post_projected_old_closure": {
                "projected_closure_rank": projected_old_closure_rank.rank,
                "combined_rank": flow_a_rank.rank,
                "intersection_dimension": flow_a_intersection,
                "mapping_closure_orthogonality_2norm": float(
                    np.linalg.norm(constrained_mapping.T @ projected_old_closure, ord=2)
                ),
                "maximum_principal_angle_to_rebuilt_closure_radians": float(
                    np.max(flow_a_angles) if len(flow_a_angles) else 0.0
                ),
            },
            "previous_unconstrained_null": {
                "source": "deterministically reconstructed from the phase-one design because the earlier JSON did not store the vector",
                "parameter_vector": previous_null_parameters.tolist(),
                "observable_vector": previous_null_coordinates.tolist(),
                "singular_values": previous_singular.tolist(),
                "asr_projection_relative_norm": previous_projection_ratio,
                "hessian_asr_maximum": float(
                    np.max(np.abs(np.sum(previous_null_hessian, axis=1)))
                ),
                "uniform_displacement_force_response_norm": float(
                    np.linalg.norm(uniform_response)
                ),
                "centered_design_residual": float(
                    np.linalg.norm(centered_unconstrained @ previous_null_parameters)
                ),
            },
            "dataset_control": control,
            "constrained_dataset": {
                "transferable_rank": constrained_transferable_rank.rank,
                "transferable_columns": centered_constrained_transferable.shape[1],
                "closure_rank": constrained_closure_rank.rank,
                "closure_columns": centered_constrained_closure.shape[1],
                "joint_rank": constrained_joint_rank.rank,
                "joint_columns": centered_constrained.shape[1],
                "joint_nullity": centered_constrained.shape[1] - constrained_joint_rank.rank,
                "singular_values": constrained_joint_rank.singular_values.tolist(),
                "rank_tolerance": constrained_joint_rank.tolerance,
                "rank_stability": dataset_rank_stability,
                "condition_number": float(
                    constrained_joint_rank.singular_values[0]
                    / constrained_joint_rank.singular_values[-1]
                ),
                "minimum_principal_angle_radians": float(
                    np.min(constrained_dataset_angles)
                ),
            },
            "force_reconstruction": {
                "transferable_only": fit_a,
                "transferable_plus_unconstrained_closure": fit_b,
                "asr_constrained_transferable_plus_closure": fit_c,
                "constrained_closure_hessian_norm_ratio": closure_norm_ratio,
                "closure_ratio_interpretation": "representation residual only; not a long-range-force fraction",
            },
            "decision": "GO" if all_go else "NO-GO",
            "reason": (
                "ASR-constrained representation and COM-removed dataset are both identifiable"
                if all_go
                else "one or more ASR-constrained representation or dataset criteria failed"
            ),
            "negative_control": aliasing_negative_control(),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=int, default=100)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    result = run(args.snapshots)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
