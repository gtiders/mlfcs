"""Native MLFCS HDF5 schema v2.

The schema stores a reference frame explicitly and labels sparse IFC entries
by primitive sites plus lattice translations.  It deliberately does not read
the pre-4.0 cell-major schema because its atom semantics are ambiguous.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from ase import Atoms

from mlfcs.core.geometry import StructureRelation
from mlfcs.model import ForceConstants, SparseOrderForceConstants

if TYPE_CHECKING:
    from mlfcs.core.geometry import PeriodicIndex


SCHEMA_VERSION = 2


def _write_atoms(group: h5py.Group, atoms: Atoms) -> None:
    group.create_dataset("cell", data=np.asarray(atoms.cell))
    group.create_dataset("positions", data=atoms.get_positions())
    group.create_dataset("numbers", data=atoms.numbers)
    group.create_dataset("pbc", data=np.asarray(atoms.pbc, dtype=bool))


def _read_atoms(group: h5py.Group) -> Atoms:
    return Atoms(
        numbers=np.asarray(group["numbers"], dtype=int),
        positions=np.asarray(group["positions"], dtype=float),
        cell=np.asarray(group["cell"], dtype=float),
        pbc=np.asarray(group["pbc"], dtype=bool),
    )


def _relation(force_constants: ForceConstants) -> StructureRelation:
    if isinstance(force_constants.relation, StructureRelation):
        return force_constants.relation
    raise ValueError(
        "native HDF5 v2 requires ForceConstants produced with an explicit structure relation"
    )


def _lattice_labels(
    values: SparseOrderForceConstants, index: PeriodicIndex
) -> tuple[np.ndarray, np.ndarray]:
    if values.is_lattice_labelled:
        assert values.sites is not None
        assert values.translation_representatives is not None
        return values.sites, values.translation_representatives
    sites = index.primitive[values.clusters]
    raw = index.translations[values.clusters[:, 1:]] - index.translations[values.clusters[:, :1]]
    translations = np.asarray(
        [[index.canonical_translation(vector) for vector in row] for row in raw], dtype=np.int32
    )
    return np.asarray(sites, dtype=np.int32), np.asarray(translations, dtype=np.int32)


def write_hdf5(target: str | Path, force_constants: ForceConstants) -> None:
    relation = _relation(force_constants)
    with h5py.File(target, "w") as handle:
        handle.attrs["format"] = "mlfcs-force-constants"
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["units"] = "eV/angstrom^order"
        handle.attrs["tensor_basis"] = "cartesian"
        structures = handle.create_group("structures")
        _write_atoms(structures.create_group("primitive"), relation.primitive)
        _write_atoms(structures.create_group("reference_supercell"), relation.reference)
        mapping = handle.create_group("reference_mapping")
        mapping.create_dataset("supercell_matrix", data=relation.supercell_matrix)
        mapping.create_dataset("primitive_index", data=relation.primitive_index)
        mapping.create_dataset("cell_translation", data=relation.cell_translation)
        mapping.attrs["maximum_position_residual_angstrom"] = relation.position_residual
        group = handle.create_group("force_constants")
        for order, values in sorted(force_constants.sparse.items()):
            entry = group.create_group(str(order))
            sites, translations = _lattice_labels(values, relation.index)
            entry.attrs["representation"] = "lattice-labelled-sparse"
            entry.attrs["order"] = order
            entry.attrs["unit"] = f"eV/angstrom^{order}"
            entry.create_dataset("sites", data=sites, compression="gzip")
            entry.create_dataset(
                "translation_representatives", data=translations, compression="gzip"
            )
            entry.create_dataset("tensors", data=values.tensors, compression="gzip")
        for key, value in force_constants.metadata.items():
            if isinstance(value, (str, int, float, bool, np.number)):
                handle.attrs[key] = value
            elif isinstance(value, dict):
                handle.attrs[key] = json.dumps(value, sort_keys=True, default=float)


def read_hdf5(source: str | Path) -> ForceConstants:
    with h5py.File(source, "r") as handle:
        if int(handle.attrs.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError("unsupported native MLFCS HDF5 schema; only v2 is supported")
        primitive = _read_atoms(handle["structures/primitive"])
        reference = _read_atoms(handle["structures/reference_supercell"])
        relation = StructureRelation.from_atoms(primitive, reference)
        mapping = handle["reference_mapping"]
        if not np.array_equal(relation.supercell_matrix, mapping["supercell_matrix"]):
            raise ValueError("HDF5 reference mapping does not match its structures")
        sparse: dict[int, SparseOrderForceConstants] = {}
        index = relation.index
        for name, entry in handle["force_constants"].items():
            order = int(name)
            sites = np.asarray(entry["sites"], dtype=np.int32)
            translations = np.asarray(entry["translation_representatives"], dtype=np.int32)
            tensors = np.asarray(entry["tensors"], dtype=float)
            clusters = np.empty_like(sites)
            clusters[:, 0] = [index.representative(int(site)) for site in sites[:, 0]]
            for axis in range(1, order):
                clusters[:, axis] = [
                    index.atom(int(site), translation)
                    for site, translation in zip(
                        sites[:, axis], translations[:, axis - 1], strict=True
                    )
                ]
            sparse[order] = SparseOrderForceConstants(
                order,
                index.n_primitive,
                len(reference),
                clusters,
                tensors,
                sites,
                translations,
            )
        metadata = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in handle.attrs.items()
            if key not in {"format", "schema_version", "units", "tensor_basis"}
        }
    return ForceConstants({}, reference, metadata=metadata, sparse=sparse, relation=relation)


__all__ = ["SCHEMA_VERSION", "read_hdf5", "write_hdf5"]
