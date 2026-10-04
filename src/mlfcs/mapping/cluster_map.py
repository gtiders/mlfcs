"""One primitive cluster model realized in a supercell."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from ase import Atoms

from mlfcs._arrays import integer_array, require_allocation
from mlfcs.cluster_space import ClusterSpace
from mlfcs.core import LatticeSite
from mlfcs.core.log import get_logger
from mlfcs.core.structure import structure_fingerprint
from mlfcs.mapping._rank import RankInfo, folded_rank
from mlfcs.mapping.geometry import (
    _PeriodicIndex,
    _quotient_kernel,
    infer_supercell_matrix,
    mapped_labels,
    prepare_periodic_index,
)

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True, init=False)
class ClusterMap:
    """Immutable realization of one ClusterSpace in one external supercell.

    Parameters
    ----------
    cluster_space : ClusterSpace
        Primitive model referenced by this mapping; it is not copied.
    supercell_atoms : ase.Atoms
        Fully periodic reference supercell. Atom order is retained and geometry
        and masses are captured independently of this ASE object.
    supercell_matrix : array_like of integers, shape (3, 3), optional
        Row-cell convention: supercell_cell = matrix @ primitive_cell. If omitted,
        infer each row by unique periodic matching at cluster_space.symprec.

    Notes
    -----
    Initialization validates the primitive-to-supercell relation and folds orbit
    images into atom indices. Arrays are readonly. A ClusterSpace may have many
    independent ClusterMap instances; this object owns one supercell realization.
    image_atom_indices[i] has shape (n_images_i, order_i). It records tensor-slot
    atom order, not a sorted set. supercell_atoms returns a detached ASE copy.

    Raises
    ------
    ValueError
        Geometry, species, atom count or periodic matching is inconsistent.
    OverflowError
        Exact quotient operations or array sizes exceed the admitted domain.
    """

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
        """Infer or validate the supercell relation and prepare quotient/image lookup buffers."""
        from mlfcs.mapping.geometry import supercell_data

        if not isinstance(cluster_space, ClusterSpace):
            raise TypeError("cluster_space must be a ClusterSpace")
        started = perf_counter()
        logger.info(
            "Mapping started: primitive_atoms=%d matrix_source=%s",
            cluster_space.n_atoms,
            "inferred" if supercell_matrix is None else "explicit",
        )
        if supercell_matrix is None:
            supercell_matrix = infer_supercell_matrix(cluster_space, supercell_atoms)
            logger.info("Supercell matrix inferred: %s", supercell_matrix.tolist())
        data = supercell_data(cluster_space, supercell_atoms, supercell_matrix)
        logger.info(
            "Supercell validated: supercell_atoms=%d matrix=%s",
            len(data["atomic_numbers"]),
            data["supercell_matrix"].tolist(),
        )
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
        logger.info(
            "Mapping complete: orbits=%d images=%d elapsed_s=%.2f",
            len(folded),
            sum(len(values) for values in folded),
            perf_counter() - started,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Mapping identity: fingerprint=%s", self.fingerprint)

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
        """Number of atoms in this reference supercell, not in the primitive motif."""
        return len(self.atomic_numbers)

    def rank_info(self, order=None) -> RankInfo:
        """Return exact structural rank and alias counts for one order or all orders.

        This evaluates folded orbit bases independently of training displacements;
        it is recomputed on each call. Missing orders raise KeyError.
        """
        started = perf_counter()
        logger.info("Folded rank started: order=%s", order if order is not None else "all")
        result = folded_rank(self.cluster_space, self.image_atom_indices, order)
        logger.info(
            "Folded rank complete: parameters=%d rank=%d nullity=%d aliases=%d elapsed_s=%.2f",
            result.parameters,
            result.rank,
            result.nullity,
            result.aliases,
            perf_counter() - started,
        )
        return result

    def __reduce__(self):
        """Serialize the primitive model and reference atoms for validated mapping reconstruction."""
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
        return tuple(
            int(v)
            for v in _quotient_kernel(values, self._periodic.adjugate, self._periodic.modulus)
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
        """Hash primitive identity, declared supercell geometry and atom-address ordering."""
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
        """Stable identity of the cluster space, supercell realization and folded image ordering."""
        payload = {
            "space": self.cluster_space.fingerprint,
            "supercell": self._geometry_fingerprint(),
            "atoms": [values.tolist() for values in self.image_atom_indices],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _restore_cluster_map(cluster_space, supercell_atoms, supercell_matrix):
    """Reconstruct and validate serialized mapping inputs through the normal constructor."""
    return ClusterMap(cluster_space, supercell_atoms, supercell_matrix=supercell_matrix)
