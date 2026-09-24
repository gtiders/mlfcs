"""Map primitive cluster images onto supercell atom indices."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from mlfcs.cluster_space import ClusterSpace
from mlfcs.cluster_space.invariants import apply_lattice_action
from mlfcs.core.errors import AliasingError
from mlfcs.supercell.cell import Supercell


@dataclass(frozen=True, slots=True)
class RankInfo:
    """Exact structural rank of a cluster map."""

    parameters: int
    rank: int
    aliases: int

    @property
    def nullity(self) -> int:
        return self.parameters - self.rank

    @property
    def full(self) -> bool:
        return self.rank == self.parameters

    def require_full(self) -> None:
        """Raise when the supercell leaves primitive parameters aliased."""
        if not self.full:
            raise AliasingError(
                f"supercell realization has rank {self.rank} for {self.parameters} "
                f"parameters (nullity {self.nullity})"
            )


@dataclass(frozen=True, slots=True)
class ClusterMap:
    """A cluster space folded into one supercell, without copied orbit objects."""

    space: ClusterSpace
    supercell: Supercell
    atoms: tuple[np.ndarray, ...]

    @classmethod
    def build(cls, space: ClusterSpace, supercell: Supercell) -> ClusterMap:
        if space.primitive.fingerprint != supercell.primitive.fingerprint:
            raise ValueError("cluster space and supercell use different primitive structures")
        lookup = {
            (int(site), *(int(value) for value in quotient)): atom
            for atom, (site, quotient) in enumerate(
                zip(supercell.sites, supercell.quotients, strict=True)
            )
        }
        folded = []
        for orbit in space.orbits:
            values = np.empty((len(orbit.clusters), orbit.representative.order), dtype=np.int32)
            for image, cluster in enumerate(orbit.clusters):
                for axis, site in enumerate(cluster.sites):
                    key = (site.site, *supercell.quotient(site.translation))
                    values[image, axis] = lookup[key]
            values.setflags(write=False)
            folded.append(values)
        return cls(space=space, supercell=supercell, atoms=tuple(folded))

    def aliases(self, order: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return folded atom tuples reached by more than one primitive image."""
        block = self.space.block(order)
        groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
        for orbit_index in range(block.orbits.start, block.orbits.stop):
            for image, atoms in enumerate(self.atoms[orbit_index]):
                groups.setdefault(tuple(int(value) for value in atoms), []).append(
                    (orbit_index, image)
                )
        return tuple(tuple(group) for _, group in sorted(groups.items()) if len(group) > 1)

    def rank_info(self, order: int | None = None) -> RankInfo:
        """Return exact structural rank, independently of training displacements."""
        if order is None:
            values = [self.rank_info(value) for value in self.space.orders]
            return RankInfo(
                parameters=sum(value.parameters for value in values),
                rank=sum(value.rank for value in values),
                aliases=sum(value.aliases for value in values),
            )
        block = self.space.block(order)
        orbit_indices = range(block.orbits.start, block.orbits.stop)
        groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
        for orbit_index in orbit_indices:
            for image, atoms in enumerate(self.atoms[orbit_index]):
                groups.setdefault(tuple(int(value) for value in atoms), []).append(
                    (orbit_index, image)
                )
        aliases = sum(len(group) - 1 for group in groups.values())
        exclusive = {group[0][0] for group in groups.values() if len(group) == 1}
        if all(orbit_index in exclusive for orbit_index in orbit_indices):
            return RankInfo(
                block.parameters.stop - block.parameters.start,
                block.parameters.stop - block.parameters.start,
                aliases,
            )

        columns = {}
        start = 0
        for orbit_index in orbit_indices:
            dimension = self.space.orbits[orbit_index].dimension
            columns[orbit_index] = start
            start += dimension
        row_entries: dict[tuple[int, ...], dict[int, int]] = {}
        for orbit_index in orbit_indices:
            orbit = self.space.orbits[orbit_index]
            column_start = columns[orbit_index]
            for image, atoms in enumerate(self.atoms[orbit_index]):
                basis = apply_lattice_action(
                    orbit.exact_lattice_basis,
                    self.space.symmetry.rotations[orbit.operations[image]],
                    tuple(int(value) for value in orbit.permutations[image]),
                )
                atom_key = tuple(int(value) for value in atoms)
                for component, values in enumerate(basis):
                    entries = row_entries.setdefault((*atom_key, component), {})
                    for column, value in enumerate(values):
                        result = entries.get(column_start + column, 0) + int(value)
                        if result:
                            entries[column_start + column] = result
                        else:
                            entries.pop(column_start + column, None)
        nonempty = [entries for entries in row_entries.values() if entries]
        locations = {
            (row, column): value
            for row, entries in enumerate(nonempty)
            for column, value in entries.items()
        }
        from sympy import SparseMatrix

        rank = int(SparseMatrix(len(nonempty), start, locations).rank())
        return RankInfo(parameters=start, rank=rank, aliases=aliases)

    @property
    def fingerprint(self) -> str:
        payload = {
            "space": self.space.fingerprint,
            "supercell": self.supercell.fingerprint,
            "atoms": [values.tolist() for values in self.atoms],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["ClusterMap", "RankInfo"]
