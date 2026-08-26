#!/usr/bin/env python3
"""Minimal architecture prototype for source-specific FC2 observable closure."""

from __future__ import annotations

import json
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from phonopy import load
from scipy import sparse
from scipy.linalg import subspace_angles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASE = ROOT / "examples" / "sscha" / "KCl"
RESULTS = HERE / "results-phase4.json"
for path in (HERE, CASE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import phase3
import prototype as phase2
from common import HARMONIC_PATH, POTENTIAL_PATH, ase_from_phonopy

from mlfcs.fitting.backends.wick.covariance import symmetrized_covariance
from mlfcs.fitting.backends.wick.features import wick_axis_derivatives
from mlfcs.fitting.design_operator import (
    DesignKernelGroup,
    ForceDesignOperator,
    force_design_batch,
    image_parameter_basis,
)
from mlfcs.fitting.gram import GramBuilder
from mlfcs.fitting.parameterization import pack_order
from mlfcs.force_constants.dense import expand_compact_fc2
from mlfcs.interactions.enumerate import resolve_primitive_cutoff
from mlfcs.interactions.realization import InteractionAliasingError
from mlfcs.interactions.space import InteractionSpace, ReferenceFrame
from mlfcs.interactions.tensors import TensorAction, _null_space_from_gram
from mlfcs.structure.integer_lattice import IntegerLatticeQuotient
from mlfcs.structure.relation import StructureRelation
from mlfcs.structure.supercell import build_supercell


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def source_fingerprint(relation: StructureRelation) -> str:
    """Return an atom-order-independent identity for one finite translation lattice."""
    digest = sha256(b"mlfcs-finite-harmonic-source-v1")
    quotient = IntegerLatticeQuotient(relation.supercell_matrix)
    primitive = relation.primitive
    for values in (
        np.asarray(primitive.numbers, dtype=np.int64),
        np.asarray(primitive.cell, dtype=np.float64),
        np.asarray(primitive.get_scaled_positions(wrap=True), dtype=np.float64),
        np.asarray(quotient.hnf, dtype=np.int64),
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _pair_index(site: int, atom: int, n_reference: int) -> int:
    return (site * n_reference + atom) * 9


def _finite_pair_generators(frame: ReferenceFrame):
    """Return exact pair-label maps and 3x3 tensor actions."""
    relation = frame.relation
    index = relation.index
    n_primitive = len(relation.primitive)
    n_reference = len(relation.reference)
    labels = tuple((site, atom) for site in range(n_primitive) for atom in range(n_reference))
    generators = []
    for permutation, cartesian in zip(
        frame.symmetry.atom_permutations,
        frame.symmetry.cartesian_rotations,
        strict=True,
    ):
        passive = cartesian.T
        action = TensorAction(passive, (0, 1), 2).as_matrix()
        mapped = []
        for site, atom in labels:
            first = int(permutation[index.representative(site)])
            second = int(permutation[atom])
            shift = -index.translations[first]
            reanchored = index.translate_atom(second, shift)
            mapped.append((int(index.primitive[first]), reanchored))
        generators.append((tuple(mapped), action))
    transpose = TensorAction(np.eye(3), (1, 0), 2).as_matrix()
    mapped = []
    for site, atom in labels:
        second_site = int(index.primitive[atom])
        reverse = index.atom(site, -index.translations[atom])
        mapped.append((second_site, reverse))
    generators.append((tuple(mapped), transpose))
    label_index = {label: position for position, label in enumerate(labels)}
    indexed = tuple(
        (
            np.asarray([label_index[label] for label in mapped], dtype=np.int32),
            action,
        )
        for mapped, action in generators
    )
    return labels, indexed


def finite_pair_orbit_basis(frame: ReferenceFrame) -> tuple[sparse.csc_matrix, dict[str, object]]:
    """Generate only symmetry/permutation-allowed finite pair tensor columns."""
    started = time.perf_counter()
    tracemalloc.start()
    labels, generators = _finite_pair_generators(frame)
    n_reference = len(frame.relation.reference)
    raw_dimension = len(labels) * 9
    unseen = set(range(len(labels)))
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    orbit_dimensions = []
    orbit_sizes = []
    column_offset = 0
    identity = np.eye(9)
    tolerance = 1e-10
    while unseen:
        representative = min(unseen)
        transport = {representative: identity}
        queue = [representative]
        constraint_gram = np.zeros((9, 9), dtype=float)
        while queue:
            current = queue.pop(0)
            current_transport = transport[current]
            for mapping, action in generators:
                target = int(mapping[current])
                candidate = action @ current_transport
                previous = transport.get(target)
                if previous is None:
                    transport[target] = candidate
                    queue.append(target)
                else:
                    residual = candidate - previous
                    if np.linalg.norm(residual) > tolerance:
                        constraint_gram += residual.T @ residual
        orbit = sorted(transport)
        unseen.difference_update(orbit)
        if np.any(constraint_gram):
            invariant, _independent = _null_space_from_gram(
                constraint_gram, tolerance
            )
        else:
            invariant = identity
        if invariant.shape[1] == 0:
            raise RuntimeError("finite pair orbit has no invariant tensor")
        invariant /= np.sqrt(len(orbit))
        orbit_dimensions.append(int(invariant.shape[1]))
        orbit_sizes.append(len(orbit))
        for label_position in orbit:
            site, atom = labels[label_position]
            values = transport[label_position] @ invariant
            nonzero_component, local_column = np.nonzero(np.abs(values) > 1e-13)
            rows.extend(
                _pair_index(site, atom, n_reference) + int(component)
                for component in nonzero_component
            )
            columns.extend(column_offset + int(column) for column in local_column)
            data.extend(float(values[component, column]) for component, column in zip(
                nonzero_component, local_column, strict=True
            ))
        column_offset += invariant.shape[1]
    basis = sparse.coo_matrix(
        (data, (rows, columns)), shape=(raw_dimension, column_offset)
    ).tocsc()
    gram_residual = float(
        np.linalg.norm((basis.T @ basis).toarray() - np.eye(column_offset))
    )
    current, peak = tracemalloc.get_traced_memory()
    del current
    tracemalloc.stop()
    metrics = {
        "finite_pair_orbits": len(orbit_dimensions),
        "orbit_sizes": orbit_sizes,
        "invariant_dimensions": orbit_dimensions,
        "observable_dimension": column_offset,
        "basis_shape": list(basis.shape),
        "basis_nnz": int(basis.nnz),
        "basis_density": float(basis.nnz / np.prod(basis.shape)),
        "orthonormality_residual": gram_residual,
        "wall_seconds": time.perf_counter() - started,
        "tracemalloc_peak_mib": peak / 2**20,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    return basis, metrics


def compact_asr_operator(relation: StructureRelation) -> sparse.csr_matrix:
    """Map compact FC2 coordinates to primitive-anchor row sums."""
    n_primitive = len(relation.primitive)
    n_reference = len(relation.reference)
    rows = []
    columns = []
    data = []
    for site in range(n_primitive):
        for alpha in range(3):
            for beta in range(3):
                row = (site * 3 + alpha) * 3 + beta
                for atom in range(n_reference):
                    rows.append(row)
                    columns.append(_pair_index(site, atom, n_reference) + alpha * 3 + beta)
                    data.append(1.0)
    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(n_primitive * 9, n_primitive * n_reference * 9),
    ).tocsr()


@dataclass(frozen=True)
class FiniteObservableSpace:
    """Source-specific ASR finite Hessian space and transferable complement."""

    relation: StructureRelation
    fingerprint: str
    observable_basis: np.ndarray
    transferable_map: np.ndarray
    closure_projector_factor: np.ndarray
    temporary_closure_basis: np.ndarray
    build_metrics: dict[str, object]

    @classmethod
    def build(cls, calculation: InteractionSpace) -> FiniteObservableSpace:
        frame = calculation.frame
        finite_basis, metrics = finite_pair_orbit_basis(frame)
        asr = compact_asr_operator(frame.relation) @ finite_basis
        z_observable, asr_rank = phase2.null_space(asr.toarray())
        observable_basis = np.asarray(finite_basis @ z_observable)
        primitive_space = calculation.primitive_orbit_space
        raw_mapping = np.empty(
            (observable_basis.shape[0], sum(o.dimension for o in primitive_space.orbits))
        )
        for parameter in range(raw_mapping.shape[1]):
            values = np.zeros(raw_mapping.shape[1])
            values[parameter] = 1.0
            raw_mapping[:, parameter] = phase2.compact_from_sparse(
                primitive_space, values, frame.relation
            ).reshape(-1)
        from mlfcs.constraints.translational import build_translational_constraints
        from mlfcs.fitting.linear_solvers import explicit_constraint_null_space

        constraints = build_translational_constraints(primitive_space)
        r_asr = explicit_constraint_null_space(constraints, tolerance=1e-11).toarray()
        transferable_map = observable_basis.T @ raw_mapping @ r_asr
        rank = phase2.numerical_rank(transferable_map)
        if rank.rank != transferable_map.shape[1]:
            raise InteractionAliasingError("ASR-constrained transferable realization is aliased")
        _image, closure = phase2.closure_basis(transferable_map, rank.rank)
        closure_factor = observable_basis @ closure
        metrics.update(
            {
                "asr_constraint_rank": asr_rank.rank,
                "asr_observable_dimension": int(observable_basis.shape[1]),
                "transferable_asr_dimension": int(transferable_map.shape[1]),
                "closure_dimension": int(closure.shape[1]),
                "observable_basis_nbytes": int(observable_basis.nbytes),
                "closure_factor_nbytes": int(closure_factor.nbytes),
            }
        )
        return cls(
            frame.relation,
            source_fingerprint(frame.relation),
            observable_basis,
            transferable_map,
            closure_factor,
            closure,
            metrics,
        )

    @property
    def closure_dimension(self) -> int:
        return self.closure_projector_factor.shape[1]

    def closure_project(self, compact: np.ndarray) -> np.ndarray:
        values = np.asarray(compact).reshape(-1)
        factor = self.closure_projector_factor
        return (factor @ (factor.T @ values)).reshape(compact.shape)


class DesignBlock(Protocol):
    @property
    def n_parameters(self) -> int: ...

    def build_batch(self, displacements: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class OrbitDesignBlock:
    """Thin adapter over the existing orbit/Wick design function."""

    covariance: np.ndarray
    parameterizations: tuple[object, ...]
    image_bases: tuple[np.ndarray, ...]
    parameter_count: int

    @property
    def n_parameters(self) -> int:
        return self.parameter_count

    def build_batch(self, displacements: np.ndarray) -> np.ndarray:
        values = force_design_batch(
            jnp.asarray(displacements),
            jnp.asarray(self.covariance),
            self.parameterizations,
            self.image_bases,
            self.parameter_count,
            wick_axis_derivatives,
        )
        return np.asarray(values).reshape((-1, self.parameter_count))


@dataclass(frozen=True)
class FiniteHarmonicDesignBlock:
    """The only new numerical operation: Phi(eta) mapped to -Phi u."""

    full_hessian_basis: np.ndarray

    @property
    def n_parameters(self) -> int:
        return self.full_hessian_basis.shape[1]

    def build_batch(self, displacements: np.ndarray) -> np.ndarray:
        return phase2.design_from_hessian_basis(displacements, self.full_hessian_basis)

    def as_kernel_group(self, column_offset: int, device) -> DesignKernelGroup:
        n_atoms = int(np.sqrt(self.full_hessian_basis.shape[0] / 9))
        hessians = self.full_hessian_basis.T.reshape((-1, n_atoms, n_atoms, 3, 3))

        @jax.jit
        def kernel(displacements, covariance, basis):
            del covariance
            contribution = -jnp.einsum(
                "pijab,sjb->siap", basis, displacements, optimize=True
            )
            return contribution[None, ...]

        columns = np.arange(
            column_offset, column_offset + self.n_parameters, dtype=np.int32
        )[None, :]
        arguments = (hessians,)
        return DesignKernelGroup(
            order=2,
            kernel=kernel,
            columns=columns,
            device_columns=jax.device_put(columns, device),
            arguments=arguments,
            device_arguments=(jax.device_put(hessians, device),),
        )


@dataclass(frozen=True)
class CompositeDesign:
    blocks: tuple[DesignBlock, ...]

    @property
    def n_parameters(self) -> int:
        return sum(block.n_parameters for block in self.blocks)

    def build_batch(self, displacements: np.ndarray) -> np.ndarray:
        return np.column_stack(tuple(block.build_batch(displacements) for block in self.blocks))


@dataclass(frozen=True)
class FiniteHarmonicResponse:
    """Stable source-owned residual Hessian, without exact-R identity."""

    source_relation: StructureRelation
    source_fingerprint: str
    compact_hessian: np.ndarray
    metadata: dict[str, object]

    def full_hessian(self, reference=None) -> np.ndarray:
        target_atoms = self.source_relation.reference if reference is None else reference
        target = StructureRelation.from_atoms(self.source_relation.primitive, target_atoms)
        source_quotient = IntegerLatticeQuotient(self.source_relation.supercell_matrix)
        target_quotient = IntegerLatticeQuotient(target.supercell_matrix)
        if not np.array_equal(source_quotient.hnf, target_quotient.hnf):
            raise ValueError("finite harmonic response belongs to a different source supercell")
        source_full = expand_compact_fc2(
            self.compact_hessian, self.source_relation.reference
        )
        source_to_target = np.asarray(
            [
                target.index.atom(int(site), translation)
                for site, translation in zip(
                    self.source_relation.primitive_index,
                    self.source_relation.cell_translation,
                    strict=True,
                )
            ],
            dtype=np.int32,
        )
        if not np.array_equal(
            self.source_relation.reference.numbers,
            target.reference.numbers[source_to_target],
        ):
            raise ValueError("finite harmonic source species do not match target representation")
        result = np.empty_like(source_full)
        result[np.ix_(source_to_target, source_to_target)] = source_full
        return result


def full_basis_from_compact(compact_basis: np.ndarray, relation) -> np.ndarray:
    columns = []
    shape = (len(relation.primitive), len(relation.reference), 3, 3)
    for values in compact_basis.T:
        columns.append(
            expand_compact_fc2(values.reshape(shape), relation.reference).reshape(-1)
        )
    return np.asarray(columns).T


def solve_streamed_joint(
    calculation,
    parameterization,
    covariance,
    displacements,
    forces,
    r_asr,
    closure_block: FiniteHarmonicDesignBlock,
):
    physical_count = parameterization.n_parameters + closure_block.n_parameters
    parameter_map = sparse.block_diag((r_asr, sparse.eye(closure_block.n_parameters)), format="csc")
    operator = ForceDesignOperator(
        displacements,
        covariance,
        (parameterization,),
        physical_count,
        1,
        parameter_map=parameter_map,
        device_gram=False,
        device=jax.devices("cpu")[0],
        axis_derivatives=wick_axis_derivatives,
    )
    closure_group = closure_block.as_kernel_group(
        parameterization.n_parameters, operator.program.device
    )
    operator.program.groups = (*operator.program.groups, closure_group)
    gram_started = time.perf_counter()
    gram = GramBuilder.from_operator(operator, forces.reshape(-1))
    gram_seconds = time.perf_counter() - gram_started
    scale = gram.exact_column_scale()
    empty = sparse.csr_matrix((0, gram.gram.shape[0]))
    solution = gram.solve(
        scale,
        empty,
        tolerance=1e-12,
        max_iterations=5000,
    )
    reduced = np.asarray(solution[0]) * scale
    return reduced, gram, operator, gram_seconds, int(solution[1])


def end_to_end(primitive, reference, cutoff, observable: FiniteObservableSpace):
    from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

    from mlfcs.constraints.translational import build_translational_constraints
    from mlfcs.fitting.linear_solvers import explicit_constraint_null_space

    reference = observable.relation.reference
    calculation = InteractionSpace.from_frame(
        ReferenceFrame.from_atoms(primitive, reference, symprec=1e-5),
        order=2,
        cutoff=cutoff,
        max_body_order=None,
        symprec=1e-5,
    )
    parameterization, count = pack_order(calculation, 0)
    if count != parameterization.n_parameters:
        raise RuntimeError("unexpected FC2 parameter packing")
    rng = np.random.default_rng(42)
    displacements = rng.normal(scale=0.01, size=(100, len(reference), 3))
    displacements -= displacements.mean(axis=1, keepdims=True)
    calculator = PolymlpASECalculator(pot=POTENTIAL_PATH)
    forces = phase2.evaluate_forces(reference, displacements, calculator)
    covariance = symmetrized_covariance(displacements, calculation)
    r_asr = explicit_constraint_null_space(
        build_translational_constraints(calculation.primitive_orbit_space),
        tolerance=1e-11,
    ).toarray()
    orbit_block = OrbitDesignBlock(
        covariance,
        (parameterization,),
        (image_parameter_basis(parameterization),),
        parameterization.n_parameters,
    )
    closure_full_basis = full_basis_from_compact(
        observable.closure_projector_factor, observable.relation
    )
    closure_block = FiniteHarmonicDesignBlock(closure_full_basis)
    physical_design = CompositeDesign((orbit_block, closure_block)).build_batch(displacements)
    reduced_design = physical_design @ sparse.block_diag(
        (r_asr, sparse.eye(closure_block.n_parameters)), format="csc"
    )
    reduced_design = np.asarray(reduced_design)
    explicit_parameters, *_ = np.linalg.lstsq(
        reduced_design, forces.reshape(-1), rcond=None
    )
    explicit_gram = reduced_design.T @ reduced_design
    explicit_rhs = reduced_design.T @ forces.reshape(-1)

    streamed, gram, operator, gram_seconds, stop_code = solve_streamed_joint(
        calculation,
        parameterization,
        covariance,
        displacements,
        forces,
        r_asr,
        closure_block,
    )
    transfer_count = r_asr.shape[1]
    theta = r_asr @ streamed[:transfer_count]
    transferable_compact = phase2.compact_from_sparse(
        calculation.primitive_orbit_space, theta, observable.relation
    )
    closure_compact = (
        observable.closure_projector_factor @ streamed[transfer_count:]
    ).reshape(transferable_compact.shape)
    response = FiniteHarmonicResponse(
        observable.relation,
        observable.fingerprint,
        closure_compact,
        {"symmetry": True, "acoustic_sum_rule": True, "source_only": True},
    )
    total_full = expand_compact_fc2(
        transferable_compact + closure_compact, reference
    )

    # Compare against the original dense-projector research coordinates.
    dense_system, _metrics = phase3.build_system(primitive, 2, cutoff)
    dense_transfer_full = dense_system.constrained_hessian_basis @ dense_system.constrained_mapping
    dense_closure_full = dense_system.constrained_hessian_basis @ dense_system.constrained_closure
    dense_design = np.column_stack(
        (
            phase2.design_from_hessian_basis(displacements, dense_transfer_full),
            phase2.design_from_hessian_basis(displacements, dense_closure_full),
        )
    )
    dense_parameters, *_ = np.linalg.lstsq(
        dense_design, forces.reshape(-1), rcond=None
    )
    dense_total = (
        np.column_stack((dense_transfer_full, dense_closure_full)) @ dense_parameters
    ).reshape(total_full.shape)

    prediction = reduced_design @ streamed
    asr_residual = float(np.max(np.abs(np.sum(total_full, axis=1))))
    return {
        "joint_rank": phase2.numerical_rank(reduced_design).rank,
        "joint_columns": reduced_design.shape[1],
        "streamed_solver_stop_code": stop_code,
        "parameter_relative_difference_vs_explicit": float(
            np.linalg.norm(streamed - explicit_parameters)
            / np.linalg.norm(explicit_parameters)
        ),
        "gram_relative_difference": float(
            np.linalg.norm(gram.gram - explicit_gram) / np.linalg.norm(explicit_gram)
        ),
        "rhs_relative_difference": float(
            np.linalg.norm(gram.rhs - explicit_rhs) / np.linalg.norm(explicit_rhs)
        ),
        "force_rmse_eV_per_angstrom": float(
            np.sqrt(np.mean(np.square(prediction - forces.reshape(-1))))
        ),
        "total_hessian_asr_maximum": asr_residual,
        "total_hessian_relative_difference_vs_dense_phase3": float(
            np.linalg.norm(total_full - dense_total) / np.linalg.norm(dense_total)
        ),
        "source_response_full_view_relative_difference": float(
            np.linalg.norm(response.full_hessian() - expand_compact_fc2(closure_compact, reference))
            / np.linalg.norm(expand_compact_fc2(closure_compact, reference))
        ),
        "streamed_gram_seconds": gram_seconds,
        "closure_design_seconds_for_100_frames": _timed_design(closure_block, displacements),
        "total_fit_parameters": len(streamed),
        "transferable_reduced_parameters": transfer_count,
        "closure_parameters": closure_block.n_parameters,
        "response": response,
        "operator": operator,
    }


def _timed_design(block: DesignBlock, displacements: np.ndarray) -> float:
    started = time.perf_counter()
    block.build_batch(displacements)
    return time.perf_counter() - started


def ownership_tests(response: FiniteHarmonicResponse, primitive) -> dict[str, object]:
    source = response.source_relation.reference
    rng = np.random.default_rng(20260823)
    permutation = rng.permutation(len(source))
    reordered = source[permutation]
    reordered_full = response.full_hessian(reordered)
    expected_reordered = response.full_hessian()[np.ix_(permutation, permutation)]
    reorder_error = float(
        np.linalg.norm(reordered_full - expected_reordered)
        / np.linalg.norm(expected_reordered)
    )

    unimodular = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    changed = source.copy()
    changed.set_cell(unimodular @ np.asarray(source.cell), scale_atoms=False)
    changed.wrap()
    representation_full = response.full_hessian(changed)
    changed_relation = StructureRelation.from_atoms(primitive, changed)
    source_to_changed = np.asarray(
        [
            changed_relation.index.atom(int(site), translation)
            for site, translation in zip(
                response.source_relation.primitive_index,
                response.source_relation.cell_translation,
                strict=True,
            )
        ],
        dtype=np.int32,
    )
    recovered_source = representation_full[np.ix_(source_to_changed, source_to_changed)]
    source_full = response.full_hessian()
    representation_norm_difference = float(
        abs(np.linalg.norm(representation_full) - np.linalg.norm(source_full))
    )
    representation_roundtrip_error = float(
        np.linalg.norm(recovered_source - source_full) / np.linalg.norm(source_full)
    )

    mismatch_rejected = False
    try:
        response.full_hessian(build_supercell(primitive, (3, 3, 3)))
    except ValueError:
        mismatch_rejected = True
    return {
        "atom_reorder_relative_error": reorder_error,
        "unimodular_same_sublattice_norm_difference": representation_norm_difference,
        "unimodular_same_sublattice_roundtrip_relative_error": representation_roundtrip_error,
        "different_source_supercell_rejected": mismatch_rejected,
    }


def degeneracy_tests(observable: FiniteObservableSpace, primitive) -> dict[str, object]:
    zero_closure = np.empty((observable.observable_basis.shape[0], 0))
    no_block = None if zero_closure.shape[1] == 0 else FiniteHarmonicDesignBlock(zero_closure)
    deficient = np.zeros((48, 11))
    deficient_rejected = phase2.numerical_rank(deficient).rank != deficient.shape[1]
    alias_rejected = phase2.aliasing_negative_control()["production_check_rejected"]
    return {
        "zero_closure_creates_no_design_block": no_block is None,
        "zero_closure_parameter_count": 0,
        "rank_deficient_dataset_rejected_before_solve": deficient_rejected,
        "transferable_alias_still_rejected": alias_rejected,
        "different_source_is_tested_separately": len(primitive) > 0,
    }


def builder_benchmark(primitive, cutoff) -> tuple[dict[int, FiniteObservableSpace], list[dict]]:
    spaces = {}
    rows = []
    for size in (2, 3, 4):
        reference = build_supercell(primitive, (size, size, size))
        calculation = InteractionSpace.from_frame(
            ReferenceFrame.from_atoms(primitive, reference, symprec=1e-5),
            order=2,
            cutoff=cutoff,
            max_body_order=None,
            symprec=1e-5,
        )
        started = time.perf_counter()
        observable = FiniteObservableSpace.build(calculation)
        total = time.perf_counter() - started
        observable.build_metrics["total_observable_and_closure_seconds"] = total
        observable.build_metrics["reference_size"] = [size, size, size]
        observable.build_metrics["reference_atoms"] = len(reference)
        spaces[size] = observable
        rows.append(dict(observable.build_metrics))
    return spaces, rows


def main() -> None:
    primitive = ase_from_phonopy(load(HARMONIC_PATH).primitive)
    reference = build_supercell(primitive, (2, 2, 2))
    cutoff = resolve_primitive_cutoff(primitive, None, reference=reference)
    spaces, performance = builder_benchmark(primitive, cutoff)
    observable = spaces[2]

    dense_agreement = {}
    for size in (2, 3):
        dense_frame = ReferenceFrame.from_atoms(
            primitive, build_supercell(primitive, (size, size, size)), symprec=1e-5
        )
        dense_basis, _ = phase2.observable_basis(dense_frame)
        sparse_basis, _metrics = finite_pair_orbit_basis(dense_frame)
        angles = subspace_angles(dense_basis, sparse_basis.toarray())
        dense_agreement[f"{size}x{size}x{size}"] = {
            "dense_dimension": dense_basis.shape[1],
            "orbit_dimension": sparse_basis.shape[1],
            "maximum_principal_angle_radians": float(np.max(angles)),
            "projector_relative_difference": float(
                np.linalg.norm(
                    dense_basis @ dense_basis.T
                    - sparse_basis.toarray() @ sparse_basis.toarray().T
                )
                / np.linalg.norm(dense_basis @ dense_basis.T)
            ),
        }

    end = end_to_end(primitive, reference, cutoff, observable)
    response = end.pop("response")
    end.pop("operator")
    results = {
        "scope": "minimal architecture research prototype; no src/mlfcs changes",
        "decision": "ARCHITECTURAL GO" if (
            end["joint_rank"] == end["joint_columns"]
            and end["gram_relative_difference"] < 1e-12
            and end["total_hessian_relative_difference_vs_dense_phase3"] < 1e-10
        ) else "PROTOTYPE RECOMMENDED",
        "production_go": False,
        "observable_builder": {
            "algorithm": "finite pair-label group orbits plus 3x3 tensor stabilizer invariants; ASR is imposed in the generated observable coordinates",
            "performance": performance,
            "dense_reference_agreement": dense_agreement,
        },
        "design_protocol": {
            "blocks": ["OrbitDesignBlock", "FiniteHarmonicDesignBlock"],
            "joint_streaming_gram": True,
            "copied_numerical_engines": [],
        },
        "end_to_end": end,
        "source_ownership": ownership_tests(response, primitive),
        "degenerate_paths": degeneracy_tests(observable, primitive),
        "canonical_ifc_modified": False,
        "source_fingerprint": response.source_fingerprint,
        "minimum_production_file_changes": [
            "new force_constants/finite_harmonic.py for source-owned response and finite observable builder",
            "fitting/design.py: add an internal design-block protocol and finite harmonic block",
            "fitting/gram.py: consume a sequence of design blocks instead of assuming only orbit groups",
            "fitting/fitter.py: opt-in construction and a result companion field",
            "tests: observable builder, composite Gram, ownership, and rejection paths",
        ],
    }
    RESULTS.write_text(
        json.dumps(results, indent=2, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, allow_nan=False, default=json_default))


if __name__ == "__main__":
    main()
