"""One primitive cluster model realized in a supercell, with prepared buffers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs._arrays import integer_array, require_allocation
from mlfcs.algebra.integer import prove_integer_product
from mlfcs.cluster_space import ClusterSpace
from mlfcs.cluster_space.preparation import PreparedClusterSpace, prepare_cluster_space
from mlfcs.core import LatticeSite
from mlfcs.core.log import get_logger
from mlfcs.core.structure import structure_fingerprint
from mlfcs.mapping._rank import RankInfo, folded_rank
from mlfcs.mapping.geometry import (
    _PeriodicIndex,
    infer_supercell_matrix,
    mapped_labels,
    prepare_periodic_index,
    quotient_kernel,
)

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True, init=False)
class ClusterMap:
    cluster_space: ClusterSpace
    supercell_matrix: np.ndarray
    cell: np.ndarray
    scaled_positions: np.ndarray
    atomic_numbers: np.ndarray
    primitive_site_indices: np.ndarray
    lattice_translations: np.ndarray
    quotient_labels: np.ndarray
    determinant: int
    _masses: np.ndarray
    image_atom_indices: tuple[np.ndarray, ...]
    _periodic: _PeriodicIndex

    def __init__(
        self, cluster_space: ClusterSpace, supercell_atoms: Atoms, *, supercell_matrix=None
    ):
        from mlfcs.mapping.geometry import supercell_data

        if not isinstance(cluster_space, ClusterSpace):
            raise TypeError("cluster_space must be a ClusterSpace")
        if supercell_matrix is None:
            supercell_matrix = infer_supercell_matrix(cluster_space, supercell_atoms)
            logger.info("Supercell matrix inferred: %s", supercell_matrix.tolist())
        data = supercell_data(cluster_space, supercell_atoms, supercell_matrix)
        object.__setattr__(self, "cluster_space", cluster_space)
        for name, value in data.items():
            object.__setattr__(self, "_masses" if name == "masses" else name, value)
        prepared = prepare_periodic_index(self)
        object.__setattr__(self, "_periodic", prepared)
        folded = []
        for orbit in cluster_space.orbits:
            require_allocation(
                "folded labels", (len(orbit.clusters) * orbit.representative.order, 4)
            )
            labels = np.asarray(
                [label for cluster in orbit.clusters for label in cluster.labels], dtype=np.int64
            )
            values = mapped_labels(labels, np.zeros((1, 3), dtype=np.int64), prepared)
            values = values.reshape(len(orbit.clusters), orbit.representative.order)
            values.setflags(write=False)
            folded.append(values)
        object.__setattr__(self, "image_atom_indices", tuple(folded))

    @property
    def supercell_atoms(self) -> Atoms:
        """A detached ASE copy of the reference supercell."""
        return Atoms(
            numbers=self.atomic_numbers,
            scaled_positions=self.scaled_positions,
            cell=self.cell,
            masses=self._masses,
            pbc=True,
        )

    @property
    def n_atoms(self):
        return len(self.atomic_numbers)

    def prepare(self, *, prepared_space=None):

        return prepare_cluster_map(self, prepared_space)

    def rank_info(self, order=None) -> RankInfo:
        return folded_rank(self.cluster_space, self.image_atom_indices, order)

    def __reduce__(self):
        return _restore_cluster_map, (
            self.cluster_space,
            self.supercell_atoms,
            self.supercell_matrix,
        )

    def quotient(self, translation: tuple[int, int, int]) -> tuple[int, int, int]:
        """Return the exact quotient label of a primitive translation."""
        values = integer_array(translation, name="lattice translation")
        if values.shape != (3,):
            raise ValueError("lattice translation must have shape (3,)")
        prove_integer_product(values.reshape(1, 3), self._periodic.adjugate)
        return tuple(
            int(v) for v in quotient_kernel(values, self._periodic.adjugate, self._periodic.modulus)
        )

    @property
    def translation_representatives(self) -> tuple[tuple[int, int, int], ...]:
        """One primitive translation per quotient, in explicit atom order for site zero."""
        seen: set[tuple[int, int, int]] = set()
        result = []
        for site, translation, quotient in zip(
            self.primitive_site_indices,
            self.lattice_translations,
            self.quotient_labels,
            strict=True,
        ):
            if int(site) != 0:
                continue
            key = tuple(int(value) for value in quotient)
            if key not in seen:
                seen.add(key)
                result.append(tuple(int(value) for value in translation))
        if len(result) != abs(self.determinant):
            raise RuntimeError("supercell lacks one translation per periodic quotient")
        return tuple(result)

    def atom_index(self, site: LatticeSite) -> int:
        """Return the supercell atom addressed by one primitive lattice site."""
        quotient = self.quotient(site.translation)
        matches = np.flatnonzero(
            (self.primitive_site_indices == site.site)
            & np.all(self.quotient_labels == quotient, axis=1)
        )
        if len(matches) != 1:
            raise RuntimeError(f"primitive site {site} has {len(matches)} supercell atoms")
        return int(matches[0])

    def aliases(self, order: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return folded atom tuples reached by more than one primitive image."""
        block = self.cluster_space.block(order)
        groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
        for orbit_index in range(block.orbits.start, block.orbits.stop):
            for image, atoms in enumerate(self.image_atom_indices[orbit_index]):
                groups.setdefault(tuple(int(value) for value in atoms), []).append(
                    (orbit_index, image)
                )
        return tuple(tuple(group) for _, group in sorted(groups.items()) if len(group) > 1)

    def _geometry_fingerprint(self) -> str:
        payload = {
            "primitive": structure_fingerprint(
                self.cluster_space.cell,
                self.cluster_space.scaled_positions,
                self.cluster_space.atomic_numbers,
                self.cluster_space.symprec,
            ),
            "matrix": self.supercell_matrix.tolist(),
            "cell": [[float(value).hex() for value in row] for row in self.cell],
            "scaled_positions": [
                [float(value).hex() for value in row] for row in self.scaled_positions
            ],
            "numbers": self.atomic_numbers.tolist(),
            "sites": self.primitive_site_indices.tolist(),
            "quotients": self.quotient_labels.tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def fingerprint(self):
        payload = {
            "space": self.cluster_space.fingerprint,
            "supercell": self._geometry_fingerprint(),
            "atoms": [values.tolist() for values in self.image_atom_indices],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _restore_cluster_map(cluster_space, supercell_atoms, supercell_matrix):
    return ClusterMap(cluster_space, supercell_atoms, supercell_matrix=supercell_matrix)


@dataclass(frozen=True, slots=True)
class PreparedClusterMap:
    fingerprint: str
    cluster_space: PreparedClusterSpace
    periodic: _PeriodicIndex
    image_atom_indices: tuple[np.ndarray, ...]


def prepare_cluster_map(cluster_map, prepared_space=None):
    space = (
        prepare_cluster_space(cluster_map.cluster_space)
        if prepared_space is None
        else prepared_space
    )
    if space.fingerprint != cluster_map.cluster_space.fingerprint:
        raise ValueError("prepared cluster space belongs to a different model")
    return PreparedClusterMap(
        cluster_map.fingerprint,
        space,
        cluster_map._periodic,
        tuple(integer_array(a) for a in cluster_map.image_atom_indices),
    )
