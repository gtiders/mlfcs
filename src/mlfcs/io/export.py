"""Strict source-to-target structure views used by all force-constant writers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.geometry import minkowski_reduce
from scipy.optimize import linear_sum_assignment

from mlfcs.core.geometry import StructureRelation
from mlfcs.model import ForceConstants, SparseOrderForceConstants


@dataclass(frozen=True, slots=True)
class ExportView:
    """A force-constant result expressed in one verified target reference frame."""

    force_constants: ForceConstants
    relation: StructureRelation | None


def _unimodular_change(target: np.ndarray, source: np.ndarray, *, name: str) -> np.ndarray:
    change = np.asarray(target) @ np.linalg.inv(np.asarray(source))
    integer = np.rint(change).astype(np.int32)
    if (
        not np.allclose(change, integer, atol=1e-7, rtol=0.0)
        or abs(round(np.linalg.det(integer))) != 1
    ):
        raise ValueError(f"target {name} is not the same lattice as the source {name}")
    return integer


def _site_mapping(source: Atoms, target: Atoms) -> tuple[np.ndarray, np.ndarray]:
    if len(source) != len(target):
        raise ValueError("target primitive atom count differs from the source primitive")
    inverse = np.linalg.inv(np.asarray(target.cell))
    mapping = np.empty(len(source), dtype=np.int32)
    shifts = np.empty((len(source), 3), dtype=np.int32)
    for number in np.unique(source.numbers):
        left = np.flatnonzero(source.numbers == number)
        right = np.flatnonzero(target.numbers == number)
        if len(left) != len(right):
            raise ValueError(
                "target primitive chemical composition differs from the source primitive"
            )
        cost = np.empty((len(left), len(right)))
        candidate_shifts = np.empty((len(left), len(right), 3), dtype=np.int32)
        for row, source_site in enumerate(left):
            for column, target_site in enumerate(right):
                # source basis point, represented in the target primitive lattice
                vector = source.positions[source_site] - target.positions[target_site]
                shift = np.rint(vector @ inverse).astype(np.int32)
                candidate_shifts[row, column] = shift
                cost[row, column] = np.linalg.norm(vector - shift @ target.cell)
        rows, columns = linear_sum_assignment(cost)
        if np.max(cost[rows, columns]) > 1e-5:
            raise ValueError("target primitive atoms are not an exactly equivalent representation")
        mapping[left[rows]] = right[columns]
        shifts[left[rows]] = candidate_shifts[rows, columns]
    return mapping, shifts


def build_export_view(
    force_constants: ForceConstants,
    *,
    primitive: Atoms | None = None,
    supercell: Atoms | None = None,
) -> ExportView:
    """Validate target structures and relabel sparse IFCs into that frame.

    No interpolation, averaging, primitive reduction, strain, or Cartesian
    rotation is accepted.  The only lattice changes allowed are integer
    unimodular basis changes, and the target supercell must retain the exact
    source translation sublattice.
    """
    if not isinstance(force_constants.relation, StructureRelation):
        if primitive is not None or supercell is not None:
            raise ValueError("target export requires force constants with a StructureRelation")
        return ExportView(force_constants, None)
    source = force_constants.relation
    target_primitive = source.primitive if primitive is None else primitive
    target_supercell = source.reference if supercell is None else supercell
    primitive_change = _unimodular_change(
        np.asarray(target_primitive.cell), np.asarray(source.primitive.cell), name="primitive"
    )
    supercell_change = _unimodular_change(
        np.asarray(target_supercell.cell), np.asarray(source.reference.cell), name="supercell"
    )
    target = StructureRelation.from_atoms(target_primitive, target_supercell)
    # Physical supercell lattices must agree after accounting for the target
    # primitive basis.  This excludes equal-volume but different sublattices.
    source_lattice_in_target = (
        supercell_change @ source.supercell_matrix @ np.linalg.inv(primitive_change)
    )
    if not np.array_equal(
        target.supercell_matrix,
        np.rint(source_lattice_in_target).astype(np.int32),
    ):
        raise ValueError("target supercell does not preserve the source translation sublattice")
    site_map, site_shift = _site_mapping(source.primitive, target.primitive)
    source_to_target_translation = np.linalg.inv(primitive_change)
    sparse: dict[int, SparseOrderForceConstants] = {}
    for order, values in force_constants.sparse.items():
        clusters = np.empty_like(values.clusters)
        labelled_sites = np.empty_like(values.clusters)
        labelled_translations = np.empty((len(values.clusters), order - 1, 3), dtype=np.int32)
        for row, cluster in enumerate(values.clusters):
            source_sites = source.index.primitive[cluster]
            source_translations = source.index.translations[cluster]
            translated = np.rint(source_translations @ source_to_target_translation).astype(
                np.int32
            )
            translated += site_shift[source_sites]
            clusters[row, 0] = target.index.representative(int(site_map[source_sites[0]]))
            labelled_sites[row, 0] = site_map[source_sites[0]]
            for axis in range(1, order):
                relative = translated[axis] - translated[0]
                clusters[row, axis] = target.index.atom(int(site_map[source_sites[axis]]), relative)
                labelled_sites[row, axis] = site_map[source_sites[axis]]
                labelled_translations[row, axis - 1] = target.index.canonical_translation(relative)
        sparse[order] = SparseOrderForceConstants(
            order,
            target.index.n_primitive,
            len(target.reference),
            clusters,
            values.tensors.copy(),
            labelled_sites,
            labelled_translations,
        )
    if force_constants.arrays and not force_constants.sparse:
        raise ValueError("target export requires lattice-labelled sparse force constants")
    converted = ForceConstants(
        {}, target.reference.copy(), dict(force_constants.metadata), sparse, target
    )
    return ExportView(converted, target)


def alamode_reduced_export_view(force_constants: ForceConstants) -> ExportView:
    """Express an IFC result in an equivalent reduced supercell basis.

    ALAMODE can encode only the 27 shifts around its supplied supercell.  A
    non-reduced but physically identical supercell can require a farther
    coefficient for its actual nearest image.  Rebase only the supercell by
    ASE's integral Minkowski operation; the primitive, atom positions, and
    physical translation sublattice remain unchanged.
    """
    if not isinstance(force_constants.relation, StructureRelation):
        raise TypeError("ALAMODE rebasing requires force constants with a StructureRelation")
    reference = force_constants.relation.reference
    reduced, _ = minkowski_reduce(reference.cell, pbc=reference.pbc)
    if np.allclose(reduced, reference.cell, atol=1e-10, rtol=0.0):
        raise ValueError("ALAMODE 27-image encoding is not improved by supercell reduction")
    target = reference.copy()
    target.set_cell(reduced, scale_atoms=False)
    return build_export_view(force_constants, supercell=target)


__all__ = ["ExportView", "alamode_reduced_export_view", "build_export_view"]
