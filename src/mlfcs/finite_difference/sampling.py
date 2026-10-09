"""Generate ordered finite-difference structures and recover force constants."""

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


def _displacement_keys(
    cluster_map: ClusterMap, order: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return the displacement coordinates required by representative observations.

    Each key contains the ``order - 1`` atomic Cartesian coordinates with
    respect to which the force is differentiated.
    """
    block = cluster_map.cluster_space.block(order)
    keys: set[tuple[tuple[int, int], ...]] = set()
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        orbit = cluster_map.cluster_space.orbits[orbit_index]
        representative_image = orbit.clusters.index(orbit.representative)
        atom_indices = tuple(
            int(value)
            for value in cluster_map.image_atom_indices[orbit_index][representative_image]
        )
        for row in orbit.observation_rows:
            directions = np.unravel_index(int(row), (3,) * order)
            keys.add(
                tuple((atom_indices[axis], int(directions[axis])) for axis in range(order - 1))
            )
    return tuple(sorted(keys))


class FiniteDifference:
    """Finite-difference sampling plan for one primitive force-constant order.

    For an order-``p`` force constant, the plan generates central-difference
    structures for the ``p - 1`` Cartesian displacement coordinates required
    by representative tensor observations.

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
    Repeated occurrences of one atom and direction are summed in the resulting
    displacement. Structures are ordered by displacement key, step length,
    then central-difference sign combination. Multiple step lengths are
    combined by extrapolating the even finite-difference error to zero step.

    The selected order must be structurally identifiable in the mapped
    supercell. Acoustic constraints are prepared by ``ClusterSpace(asr=True)``
    and applied during ``reap()`` without changing the sampling sequence.

    Raises
    ------
    ValueError
        Steps or order are invalid.
    AliasingError
        Supercell folding loses required parameters.
    """

    __slots__ = ("_displacement_keys", "_sign_combinations", "cluster_map", "disps", "order")

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
        step_lengths = tuple(sorted(float(step_length) for step_length in source))
        if not step_lengths:
            raise ValueError("finite differences require at least one step")
        if any(not np.isfinite(step_length) or step_length <= 0.0 for step_length in step_lengths):
            raise ValueError("finite-difference step lengths must be positive finite lengths")
        if len(set(step_lengths)) != len(step_lengths):
            raise ValueError("finite-difference step lengths must be distinct")
        cluster_map.cluster_space.block(order)
        cluster_map.rank_info(order).require_full()
        self.cluster_map = cluster_map
        self.order = order
        self.disps = step_lengths
        self._displacement_keys = _displacement_keys(cluster_map, order)
        self._sign_combinations = np.asarray(
            list(product((-1, 1), repeat=order - 1)), dtype=np.int8
        )
        self._sign_combinations.setflags(write=False)
        logger.info(
            "Prepared FC%d finite difference: %d configurations at step lengths %s Å",
            order,
            self.n_configurations,
            ", ".join(f"{step_length:.10g}" for step_length in self.disps),
        )

    @property
    def n_configurations(self) -> int:
        """Number of displaced structures in the canonical sampling sequence."""
        return len(self._displacement_keys) * len(self.disps) * len(self._sign_combinations)

    def sow(self) -> Sequence[Atoms]:
        """Generate displaced ASE structures lazily in canonical sampling order."""
        return _DisplacedStructures(self)

    def reap(self, dataset: ForceDataset) -> ForceConstants:
        """Reconstruct this force-constant order from an ordered force dataset.

        Frames must correspond one-to-one with ``sow()`` in canonical
        sampling order. Reconstruction uses the stored forces and does not use
        frame geometry or metadata to infer or repair their order. If the mapped
        ``ClusterSpace`` has acoustic coordinates for this order, reconstructed
        physical parameters are projected onto that subspace by least squares.
        The returned ``ForceConstants`` contains only this order.
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


class _DisplacedStructures(Sequence[Atoms]):
    """Lazy sequence of displaced structures defined by a finite-difference plan."""

    __slots__ = ("_finite_difference",)

    def __init__(self, finite_difference: FiniteDifference):
        """Create a lazy structure view for one finite-difference plan."""
        self._finite_difference = finite_difference

    def __len__(self) -> int:
        """Return the number of structures required by the sampling plan."""
        return self._finite_difference.n_configurations

    def __getitem__(self, index: int | slice) -> Atoms | tuple[Atoms, ...]:
        """Return freshly generated displaced structures by index or slice.

        Negative indices follow sequence conventions. Each structure is an
        independent copy of the mapped reference supercell and carries no
        sampling metadata or force results. Invalid indices raise ``IndexError``.
        """
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("structure index is outside the finite-difference sampling sequence")
        plan = self._finite_difference
        sign_count = len(plan._sign_combinations)
        step_count = len(plan.disps)
        key_index, remainder = divmod(position, step_count * sign_count)
        step_index, sign_index = divmod(remainder, sign_count)
        key = plan._displacement_keys[key_index]
        step_length = plan.disps[step_index]
        displacements = np.zeros((len(plan.cluster_map.atomic_numbers), 3))
        for sign, (atom, axis) in zip(plan._sign_combinations[sign_index], key, strict=True):
            displacements[int(atom), int(axis)] += int(sign) * step_length
        atoms = plan.cluster_map.supercell_atoms
        atoms.positions += displacements
        return atoms

    def __iter__(self) -> Iterator[Atoms]:
        """Yield fresh structures in canonical key, step-length, then sign order."""
        for index in range(len(self)):
            yield self[index]


__all__ = ["FiniteDifference"]
