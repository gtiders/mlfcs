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


def design_from_hessian_basis(displacements: np.ndarray, hessian_basis: np.ndarray) -> np.ndarray:
    n_atoms = displacements.shape[1]
    hessians = hessian_basis.T.reshape((-1, n_atoms, n_atoms, 3, 3))
    forces = -np.einsum("pijab,sjb->siap", hessians, displacements, optimize=True)
    return forces.reshape((-1, len(hessians)))


def kcl_structures(snapshots: int) -> tuple[Atoms, Atoms, np.ndarray, np.ndarray]:
    phonon = load(HARMONIC_PATH)
    primitive = ase_from_phonopy(phonon.primitive)
    reference = build_supercell(primitive, (2, 2, 2))
    rng = np.random.default_rng(42)
    displacements = rng.normal(scale=0.01, size=(snapshots, len(reference), 3))
    displacements -= displacements.mean(axis=1, keepdims=True)
    from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

    calculator = PolymlpASECalculator(pot=POTENTIAL_PATH)
    forces = np.empty_like(displacements)
    for snapshot, displacement in enumerate(displacements):
        atoms = reference.copy()
        atoms.positions += displacement
        atoms.calc = calculator
        forces[snapshot] = atoms.get_forces()
    return primitive, reference, displacements, forces


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
    primitive, reference, displacements, forces = kcl_structures(snapshots)
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

    image, closure = closure_basis(mapping, map_rank.rank)
    combined = np.column_stack((mapping, closure))
    combined_rank = numerical_rank(combined)
    orthogonality = float(np.linalg.norm(mapping.T @ closure, ord=2))
    intersection_angle = float(np.min(subspace_angles(image, closure)))

    rng = np.random.default_rng(20260823)
    target = rng.normal(size=finite_basis.shape[1])
    recovered, *_ = np.linalg.lstsq(combined, target, rcond=None)
    reconstructed = combined @ recovered
    coordinate_error = float(np.linalg.norm(reconstructed - target) / np.linalg.norm(target))

    hessian_basis = full_hessian_basis(finite_basis, frame.relation)
    target_hessian = hessian_basis @ target
    recovered_hessian = hessian_basis @ reconstructed
    hessian_error = float(
        np.linalg.norm(recovered_hessian - target_hessian) / np.linalg.norm(target_hessian)
    )

    finite_design = design_from_hessian_basis(displacements, hessian_basis)
    transferable_design = finite_design @ mapping
    closure_design = finite_design @ closure
    joint_design = np.column_stack((transferable_design, closure_design))
    transferable_rank = numerical_rank(transferable_design)
    closure_rank = numerical_rank(closure_design)
    joint_rank = numerical_rank(joint_design)
    fitted, *_ = np.linalg.lstsq(joint_design, forces.reshape(-1), rcond=None)
    prediction = joint_design @ fitted
    force_rmse = float(np.sqrt(np.mean(np.square(prediction - forces.reshape(-1)))))
    transferable_fit, *_ = np.linalg.lstsq(
        transferable_design, forces.reshape(-1), rcond=None
    )
    transferable_prediction = transferable_design @ transferable_fit
    transferable_force_rmse = float(
        np.sqrt(np.mean(np.square(transferable_prediction - forces.reshape(-1))))
    )
    dataset_angles = subspace_angles(transferable_design, closure_design)
    minimum_angle = float(np.min(dataset_angles)) if len(dataset_angles) else float("nan")
    condition = (
        float(joint_rank.singular_values[0] / joint_rank.singular_values[joint_rank.rank - 1])
        if joint_rank.rank
        else float("inf")
    )
    _left, _singular, right = np.linalg.svd(joint_design, full_matrices=False)
    dataset_nullity = joint_design.shape[1] - joint_rank.rank
    null_design_residual = 0.0
    null_asr_maximum = 0.0
    if dataset_nullity:
        null_parameters = right[joint_rank.rank :].T[:, 0]
        null_coordinates = combined @ null_parameters
        null_hessian = (hessian_basis @ null_coordinates).reshape(
            (len(reference), len(reference), 3, 3)
        )
        null_design_residual = float(np.linalg.norm(joint_design @ null_parameters))
        null_asr_maximum = float(np.max(np.abs(np.sum(null_hessian, axis=1))))

    full_column_rank = joint_rank.rank == joint_design.shape[1]
    result.update(
        {
            "combined_rank": combined_rank.rank,
            "combined_columns": combined.shape[1],
            "mapping_closure_orthogonality_2norm": orthogonality,
            "minimum_representation_principal_angle_radians": intersection_angle,
            "random_coordinate_reconstruction_relative_error": coordinate_error,
            "random_hessian_reconstruction_relative_error": hessian_error,
            "transferable_dataset_rank": transferable_rank.rank,
            "closure_dataset_rank": closure_rank.rank,
            "joint_dataset_rank": joint_rank.rank,
            "joint_dataset_columns": joint_design.shape[1],
            "joint_dataset_nullity": dataset_nullity,
            "joint_dataset_rank_tolerance": joint_rank.tolerance,
            "joint_dataset_condition_number": condition,
            "minimum_dataset_principal_angle_radians": minimum_angle,
            "dataset_null_design_residual": null_design_residual,
            "dataset_null_hessian_asr_maximum": null_asr_maximum,
            "transferable_only_force_fit_rmse_eV_per_angstrom": transferable_force_rmse,
            "actual_force_fit_rmse_eV_per_angstrom": force_rmse,
            "decision": "GO" if full_column_rank else "NO-GO",
            "reason": (
                "sweet spot exists and the sampled joint design has full column rank"
                if full_column_rank
                else "actual displacement data do not identify the combined basis"
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
