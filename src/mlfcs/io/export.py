"""Strict source-to-target structure views used by all force-constant writers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import numpy as np
from ase import Atoms
from ase.geometry import minkowski_reduce
from scipy.optimize import linear_sum_assignment

from mlfcs.core.geometry import StructureRelation
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants


@dataclass(frozen=True, slots=True)
class ExportView:
    """A force-constant result expressed in one verified target reference frame."""

    force_constants: ForceConstants
    relation: StructureRelation | None


def _atoms_fingerprint(atoms: Atoms | None) -> str | None:
    if atoms is None:
        return None
    digest = sha256()
    for values in (
        np.asarray(atoms.numbers, dtype=np.int64),
        np.asarray(atoms.cell, dtype=np.float64),
        np.asarray(atoms.positions, dtype=np.float64),
        np.asarray(atoms.pbc, dtype=np.uint8),
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _export_cache_key(force_constants: ForceConstants, primitive, supercell) -> tuple[object, ...]:
    sparse_identity = tuple(
        (
            int(order),
            id(values.tensors),
            id(values.sites),
            id(values.translations),
        )
        for order, values in sorted(force_constants.sparse.items())
    )
    return (
        id(force_constants.relation),
        _atoms_fingerprint(primitive),
        _atoms_fingerprint(supercell),
        sparse_identity,
    )


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
    rotation is accepted.  Primitive basis changes must be integer
    unimodular.  The target may be any verified integer supercell of that
    primitive; exact real-space labels are folded only in the target view.
    """
    cache_key = _export_cache_key(force_constants, primitive, supercell)
    cached = force_constants._export_view_cache.get(cache_key)
    if cached is not None:
        print("MLFCS export view: cache hit; reusing existing view", flush=True)
        return cached
    print("MLFCS export view: cache miss; constructing new view", flush=True)
    construction_started = perf_counter()
    if not isinstance(force_constants.relation, StructureRelation):
        if primitive is not None or supercell is not None:
            raise ValueError("target export requires force constants with a StructureRelation")
        view = ExportView(force_constants, None)
        force_constants._export_view_cache[cache_key] = view
        print(
            f"MLFCS export view: constructed in {perf_counter() - construction_started:.6f} s",
            flush=True,
        )
        return view
    source = force_constants.relation
    target_primitive = source.primitive if primitive is None else primitive
    target_supercell = source.reference if supercell is None else supercell
    primitive_change = _unimodular_change(
        np.asarray(target_primitive.cell), np.asarray(source.primitive.cell), name="primitive"
    )
    target = StructureRelation.from_atoms(target_primitive, target_supercell)
    site_map, site_shift = _site_mapping(source.primitive, target.primitive)
    source_to_target_translation = np.linalg.inv(primitive_change)
    sparse: dict[int, SparseOrderForceConstants] = {}
    for order, values in force_constants.sparse.items():
        labelled_sites = np.empty_like(values.sites)
        labelled_translations = np.empty_like(values.translations)
        for row, (source_sites, source_relative) in enumerate(
            zip(values.sites, values.translations, strict=True)
        ):
            source_translations = np.vstack((np.zeros((1, 3), dtype=np.int32), source_relative))
            translated = np.rint(source_translations @ source_to_target_translation).astype(
                np.int32
            )
            translated += site_shift[source_sites]
            labelled_sites[row, 0] = site_map[source_sites[0]]
            for axis in range(1, order):
                relative = translated[axis] - translated[0]
                labelled_sites[row, axis] = site_map[source_sites[axis]]
                labelled_translations[row, axis - 1] = relative
        sparse[order] = SparseOrderForceConstants(
            order,
            labelled_sites,
            labelled_translations,
            values.tensors.copy(),
        )
    if force_constants.arrays and not force_constants.sparse:
        raise ValueError("target export requires lattice-labelled sparse force constants")
    converted = ForceConstants(
        {}, target.reference.copy(), dict(force_constants.metadata), sparse, target
    )
    view = ExportView(converted, target)
    force_constants._export_view_cache[cache_key] = view
    print(
        f"MLFCS export view: constructed in {perf_counter() - construction_started:.6f} s",
        flush=True,
    )
    return view


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
