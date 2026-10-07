"""Define ordered finite-difference structures and reconstruct their force derivatives."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import product
from time import perf_counter

import numpy as np
from ase import Atoms

from mlfcs.dataset import ForceDataset
from mlfcs.finite_difference.reconstruction import reconstruct_force_constants
from mlfcs.force_constants import ForceConstants
from mlfcs.foundation.log import get_logger
from mlfcs.mapping import ClusterMap

logger = get_logger(__name__)


def _keys(cluster_map: ClusterMap, order: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Find the distinct Cartesian displacements needed by representative observations.

    The final atom index is the force component; the preceding indices identify
    the displaced atoms and Cartesian directions.
    """
    block = cluster_map.cluster_space.block(order)
    keys: set[tuple[tuple[int, int], ...]] = set()
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        orbit = cluster_map.cluster_space.orbits[orbit_index]
        image = orbit.clusters.index(orbit.representative)
        atoms = tuple(int(value) for value in cluster_map.image_atom_indices[orbit_index][image])
        for row in orbit.observation_rows:
            directions = np.unravel_index(int(row), (3,) * order)
            keys.add(tuple((atoms[axis], int(directions[axis])) for axis in range(order - 1)))
    return tuple(sorted(keys))


class FiniteDifference:
    """Define finite-difference force measurements for one primitive force-constant order.

    Parameters
    ----------
    cluster_map : ClusterMap
        Reference supercell relation, retained by reference.
    order : int
        Included force-constant tensor order, at least two.
    disps : float or sequence of float, default 0.01
        Distinct positive step lengths in angstrom. Sorted steps are combined
        by extrapolating the even central-difference error to zero step.

    Notes
    -----
    The plan differentiates forces with respect to order minus one Cartesian
    displacements. Repeating an atom/direction in one stencil adds its signed
    displacement. Samples are ordered by displacement key, ascending step,
    then sign combination. Multiple steps extrapolate the even central-difference
    error to zero step. The recovered order-p coefficients use ASE energy units
    per angstrom to the power p.

    Construction requires a structurally full-rank supercell mapping but does
    not call a calculator. External calculations are collected through
    ForceDataset before reconstruction.

    Raises
    ------
    ValueError
        Steps or order are invalid.
    AliasingError
        Supercell folding loses required parameters.
    """

    __slots__ = ("_keys", "_signs", "cluster_map", "disps", "order")

    def __init__(
        self,
        cluster_map: ClusterMap,
        *,
        order: int,
        disps: float | Sequence[float] = 0.01,
    ):
        """Validate the sampling domain and prepare its canonical displacement stencil."""
        order = int(order)
        if order < 2:
            raise ValueError("force-constant order must be at least 2")
        source = (disps,) if np.isscalar(disps) else disps
        values = tuple(sorted(float(disp) for disp in source))
        if not values:
            raise ValueError("finite differences require at least one step")
        if any(not np.isfinite(disp) or disp <= 0.0 for disp in values):
            raise ValueError("finite-difference displacements must be positive finite lengths")
        if len(set(values)) != len(values):
            raise ValueError("finite-difference displacements must be distinct")
        cluster_map.cluster_space.block(order)
        cluster_map.rank_info(order).require_full()
        self.cluster_map = cluster_map
        self.order = order
        self.disps = values
        self._keys = _keys(cluster_map, order)
        self._signs = np.asarray(list(product((-1, 1), repeat=order - 1)), dtype=np.int8)
        self._signs.setflags(write=False)
        logger.info(
            "Prepared FC%d finite difference: %d configurations at displacements %s Å",
            order,
            self.n_configurations,
            ", ".join(f"{disp:.10g}" for disp in self.disps),
        )

    @property
    def n_configurations(self) -> int:
        """Number of structures in the key, step, and sign-product sampling sequence."""
        return len(self._keys) * len(self.disps) * len(self._signs)

    def displacements(self) -> Sequence[Atoms]:
        """Return the lazy sequence of structures in canonical sampling order."""
        return Displacements(self)

    def reconstruct(self, dataset: ForceDataset) -> ForceConstants:
        """Recover this order from a ForceDataset in canonical sampling order.

        Only the sample count and force shape are checked. The caller must
        preserve the order produced by displacements(); no geometry matching,
        metadata lookup, calculator evaluation or reordering is performed.
        """
        started = perf_counter()
        logger.info(
            "Finite-difference reconstruction started: order=%d configurations=%d steps_angstrom=%s",
            self.order,
            self.n_configurations,
            self.disps,
        )
        model = reconstruct_force_constants(self, dataset)
        logger.info(
            "Finite-difference reconstruction complete: order=%d parameters=%d elapsed_s=%.2f",
            self.order,
            len(model.coefficients[self.order]),
            perf_counter() - started,
        )
        return model


class Displacements(Sequence[Atoms]):
    """Sequence view that lazily creates structures for one sampling plan."""

    __slots__ = ("_finite_difference",)

    def __init__(self, finite_difference: FiniteDifference):
        """Retain a sampling plan and defer construction of its ASE structures."""
        self._finite_difference = finite_difference

    def __len__(self) -> int:
        """Return the number of structures required by the sampling plan."""
        return self._finite_difference.n_configurations

    def __getitem__(self, index: int | slice) -> Atoms | tuple[Atoms, ...]:
        """Generate fresh ASE structures by integer index or slice.

        Negative indices follow sequence conventions. Structures carry no
        sampling metadata or force results. Invalid indices raise IndexError.
        """
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("displacement index is outside the finite difference")
        difference = self._finite_difference
        sign_count = len(difference._signs)
        disp_count = len(difference.disps)
        key_index, remainder = divmod(position, disp_count * sign_count)
        disp_index, sign_index = divmod(remainder, sign_count)
        key = difference._keys[key_index]
        disp = difference.disps[disp_index]
        delta = np.zeros((len(difference.cluster_map.atomic_numbers), 3))
        for sign, (atom, axis) in zip(difference._signs[sign_index], key, strict=True):
            delta[int(atom), int(axis)] += int(sign) * disp
        atoms = difference.cluster_map.supercell_atoms
        atoms.positions += delta
        return atoms

    def __iter__(self) -> Iterator[Atoms]:
        """Yield fresh structures in canonical key, step, then sign order."""
        for index in range(len(self)):
            yield self[index]


__all__ = ["FiniteDifference"]
