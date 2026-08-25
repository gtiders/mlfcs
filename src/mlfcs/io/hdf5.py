"""Native MLFCS HDF5 schema v3: primitive structure plus exact real-space IFCs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms

from mlfcs.force_constants.representation import ForceConstants, SparseOrderForceConstants
from mlfcs.structure.relation import StructureRelation

SCHEMA_VERSION = 3


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
        "native HDF5 v3 requires ForceConstants produced with an explicit structure relation"
    )


def write_hdf5(target: str | Path, force_constants: ForceConstants) -> None:
    relation = _relation(force_constants)
    with h5py.File(target, "w") as handle:
        handle.attrs["format"] = "mlfcs-force-constants"
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["units"] = "eV/angstrom^order"
        handle.attrs["tensor_basis"] = "cartesian"
        structures = handle.create_group("structures")
        _write_atoms(structures.create_group("primitive"), relation.primitive)
        if force_constants.periodic_fc2_completion is not None:
            _write_atoms(
                structures.create_group("periodic_fc2_source"), relation.reference
            )
        group = handle.create_group("force_constants")
        for order, values in sorted(force_constants.sparse.items()):
            entry = group.create_group(str(order))
            entry.attrs["representation"] = "lattice-labelled-sparse"
            entry.attrs["order"] = order
            entry.attrs["unit"] = f"eV/angstrom^{order}"
            entry.create_dataset("sites", data=values.sites, compression="gzip")
            entry.create_dataset("translations", data=values.translations, compression="gzip")
            entry.create_dataset("tensors", data=values.tensors, compression="gzip")
        if force_constants.periodic_fc2_completion is not None:
            completion = force_constants.periodic_fc2_completion
            entry = handle.create_group("periodic_fc2_completion")
            entry.attrs["representation"] = "source-periodic-harmonic-hessian"
            entry.attrs["source_fingerprint"] = completion.source_fingerprint
            entry.attrs["rank_diagnostics"] = json.dumps(
                asdict(completion.rank_report), sort_keys=True
            )
            entry.create_dataset(
                "compact_hessian", data=completion.compact_hessian, compression="gzip"
            )
        for key, value in force_constants.metadata.items():
            if isinstance(value, (str, int, float, bool, np.number)):
                handle.attrs[key] = value
            elif isinstance(value, dict):
                handle.attrs[key] = json.dumps(value, sort_keys=True, default=float)


def read_hdf5(source: str | Path) -> ForceConstants:
    """Read canonical primitive exact-R force constants from native HDF5 v3."""
    with h5py.File(source, "r") as handle:
        if int(handle.attrs.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError("unsupported native MLFCS HDF5 schema; only v3 is supported")
        primitive = _read_atoms(handle["structures/primitive"])
        # Canonical exact-R storage has no source-supercell identity.  The
        # optional periodic FC2 sidecar is source-bound and therefore carries
        # its reference structure explicitly.
        reference = (
            _read_atoms(handle["structures/periodic_fc2_source"])
            if "periodic_fc2_source" in handle["structures"]
            else primitive
        )
        relation = StructureRelation.from_atoms(primitive, reference)
        sparse: dict[int, SparseOrderForceConstants] = {}
        for name, entry in handle["force_constants"].items():
            order = int(name)
            sites = np.asarray(entry["sites"], dtype=np.int32)
            translations = np.asarray(entry["translations"], dtype=np.int32)
            tensors = np.asarray(entry["tensors"], dtype=float)
            sparse[order] = SparseOrderForceConstants(order, sites, translations, tensors)
        metadata = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in handle.attrs.items()
            if key not in {"format", "schema_version", "units", "tensor_basis"}
        }
        completion = None
        if "periodic_fc2_completion" in handle:
            from mlfcs.force_constants.periodic_fc2 import (
                PeriodicFC2Completion,
                PeriodicFC2RankReport,
            )

            entry = handle["periodic_fc2_completion"]
            rank_report = PeriodicFC2RankReport(
                **json.loads(entry.attrs["rank_diagnostics"])
            )
            completion = PeriodicFC2Completion(
                relation,
                np.asarray(entry["compact_hessian"], dtype=float),
                rank_report,
            )
            stored_fingerprint = str(entry.attrs["source_fingerprint"])
            if completion.source_fingerprint != stored_fingerprint:
                raise ValueError("periodic FC2 source fingerprint is inconsistent")
    return ForceConstants(
        {},
        relation.reference.copy(),
        metadata=metadata,
        sparse=sparse,
        relation=relation,
        periodic_fc2_completion=completion,
    )


__all__ = ["SCHEMA_VERSION", "read_hdf5", "write_hdf5"]
