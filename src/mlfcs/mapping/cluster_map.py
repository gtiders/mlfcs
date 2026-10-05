"""One primitive cluster model realized in a supercell."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from ase import Atoms

from mlfcs._arrays import as_int64_array, require_allocation
from mlfcs.cluster_space import ClusterSpace
from mlfcs.geometry.primitive import LatticeSite
from mlfcs.log import get_logger
from mlfcs.mapping.folding import RankInfo, folded_rank, group_folded_images
from mlfcs.mapping.periodic import (
    PeriodicIndex,
    map_labels,
    prepare_periodic_index,
    quotient_kernel,
)
from mlfcs.mapping.supercell import infer_supercell_matrix, prepare_supercell_data

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
    _periodic: PeriodicIndex

    def __init__(
        self, cluster_space: ClusterSpace, supercell_atoms: Atoms, *, supercell_matrix=None
    ):
        """Infer or validate the supercell relation and prepare quotient/image lookup buffers."""
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
        data = prepare_supercell_data(cluster_space, supercell_atoms, supercell_matrix)
        logger.info(
            "Supercell validated: supercell_atoms=%d matrix=%s",
            len(data["atomic_numbers"]),
            data["supercell_matrix"].tolist(),
        )
        object.__setattr__(self, "cluster_space", cluster_space)
        for name, value in data.items():
            object.__setattr__(self, "_masses" if name == "masses" else name, value)
        prepared = prepare_periodic_index(
            data["supercell_matrix"],
            data["determinant"],
            data["primitive_site_indices"],
            data["quotient_labels"],
        )
        object.__setattr__(self, "_periodic", prepared)
        folded = []
        for orbit in cluster_space.orbits:
            require_allocation(
                "folded labels", (len(orbit.clusters) * orbit.representative.order, 4)
            )
            labels = np.asarray(
                [label for cluster in orbit.clusters for label in cluster.labels], dtype=np.int64
            )
            values = map_labels(labels, np.zeros((1, 3), dtype=np.int64), prepared)
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
        """Return structural rank and alias counts for one order or all orders.

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

    def quotient(self, translation: tuple[int, int, int]) -> tuple[int, int, int]:
        """Return the quotient label of a primitive translation."""
        values = as_int64_array(translation, name="lattice translation")
        if values.shape != (3,):
            raise ValueError("lattice translation must have shape (3,)")
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
        label = np.asarray([[site.site, *site.translation]], dtype=np.int64)
        try:
            return int(self.map_labels(label, np.zeros((1, 3), dtype=np.int64))[0, 0])
        except ValueError as error:
            raise RuntimeError(f"primitive site {site} is absent from the supercell") from error

    def map_labels(self, labels: object, translations: object) -> np.ndarray:
        """Map primitive lattice labels and translation offsets to atom indices.

        Labels have shape (n, 4) with columns (site, tx, ty, tz); translations
        have shape (m, 3). Return int64 shape (n, m), preserving both input
        orders. Actual integer additions and quotient operations are checked in
        the compiled kernel; malformed shapes raise ValueError and unsafe
        intermediates raise OverflowError.
        """
        return map_labels(labels, translations, self._periodic)

    def aliases(self, order: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return folded atom tuples reached by more than one primitive image."""
        block = self.cluster_space.block(order)
        groups = group_folded_images(
            self.image_atom_indices, range(block.orbits.start, block.orbits.stop)
        )
        return tuple(tuple(group) for _, group in sorted(groups.items()) if len(group) > 1)
