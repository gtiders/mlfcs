"""Immutable primitive-cell cluster-space models."""

from __future__ import annotations

import operator
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs._arrays import require_bound
from mlfcs.cluster_space.records import Orbit, OrderBlock
from mlfcs.geometry.primitive import primitive_data
from mlfcs.geometry.symmetry import PrimitiveSymmetry
from mlfcs.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True, init=False)
class ClusterSpace:
    """Immutable primitive geometry, symmetry orbits and force-constant parameters.

    Parameters
    ----------
    primitive_atoms : ase.Atoms
        Fully periodic primitive reference structure. Atom order and cell basis
        are retained; geometry and masses are captured independently of the input.
    cutoffs : mapping of int to float
        Included tensor orders (at least two) and positive pairwise cutoffs in
        angstrom. Every pair of sites in a retained cluster lies below its cutoff.
    max_body_orders : mapping of int to int, optional
        Maximum distinct lattice sites per order. Must cover the same orders as
        cutoffs; omitted values default to each tensor order.
    symprec : float, default 1e-5
        Positive Cartesian symmetry matching tolerance in angstrom.

    Notes
    -----
    Initialization discovers symmetry, enumerates clusters, builds invariant
    bases and assigns contiguous parameter slices. This can be expensive.
    Geometry and basis buffers are readonly. No supercell, fitted coefficients
    or training structures are owned; ClusterMap supplies each supercell relation.
    The primitive_atoms property creates a detached ASE object.
    masses exposes readonly per-site atomic masses; with_masses replaces only
    their assignment in a new space sharing the existing structural model.

    Raises
    ------
    ValueError
        Primitive geometry, truncation inputs or symmetry matching are invalid.
    OverflowError
        A local integer or array-size contract cannot be satisfied.

    Examples
    --------
    >>> cs = ClusterSpace(atoms, cutoffs={2: 5.0, 3: 3.0})
    >>> cs.orders
    (2, 3)
    """

    cell: np.ndarray
    scaled_positions: np.ndarray
    atomic_numbers: np.ndarray
    symprec: float
    _masses: np.ndarray
    symmetry: PrimitiveSymmetry
    blocks: tuple[OrderBlock, ...]
    orbits: tuple[Orbit, ...]

    def __init__(self, primitive_atoms: Atoms, *, cutoffs, max_body_orders=None, symprec=1e-5):
        """Capture validated primitive data and construct the requested order blocks and orbits."""
        from mlfcs.cluster_space.construction import construct_space

        geometry = primitive_data(primitive_atoms, symprec)
        cutoffs = {operator.index(order): float(cutoff) for order, cutoff in cutoffs.items()}
        bodies = (
            {order: order for order in cutoffs}
            if max_body_orders is None
            else {
                operator.index(order): operator.index(body)
                for order, body in max_body_orders.items()
            }
        )
        symmetry, blocks, orbits = construct_space(
            geometry["cell"],
            geometry["scaled_positions"],
            geometry["atomic_numbers"],
            geometry["symprec"],
            cutoffs=cutoffs,
            max_body_orders=bodies,
        )
        for name in ("cell", "scaled_positions", "atomic_numbers", "symprec"):
            object.__setattr__(self, name, geometry[name])
        object.__setattr__(self, "_masses", geometry["masses"])
        for name, value in (
            ("symmetry", symmetry),
            ("blocks", blocks),
            ("orbits", orbits),
        ):
            object.__setattr__(self, name, value)
        self.__post_init__()

    @property
    def primitive_atoms(self) -> Atoms:
        """A detached ASE copy of the primitive reference structure."""
        return Atoms(
            numbers=self.atomic_numbers,
            scaled_positions=self.scaled_positions,
            cell=self.cell,
            masses=self._masses,
            pbc=True,
        )

    @property
    def masses(self) -> np.ndarray:
        """Readonly primitive-site masses, shape (n_atoms,), in atomic mass units."""
        return self._masses

    def with_masses(self, masses: object) -> ClusterSpace:
        """Return a new mass assignment sharing all immutable structural data.

        masses must be finite, strictly positive and have shape (n_atoms,).
        Values are copied; changing the source does not affect either space.
        No symmetry discovery, cluster enumeration or kernel solve is repeated.
        The physical parameter layout is unchanged. Rebind existing physical
        coefficients with ForceConstants(new_space, model.coefficients).
        Invalid shape or values raise ValueError.
        """
        values = np.array(masses, dtype=np.float64, copy=True, order="C")
        if values.shape != (self.n_atoms,) or not np.all(np.isfinite(values)):
            raise ValueError("masses must be finite and have shape (n_atoms,)")
        if np.any(values <= 0.0):
            raise ValueError("masses must be strictly positive")
        values.setflags(write=False)
        result = object.__new__(type(self))
        for name in (
            "cell",
            "scaled_positions",
            "atomic_numbers",
            "symprec",
            "symmetry",
            "blocks",
            "orbits",
        ):
            object.__setattr__(result, name, getattr(self, name))
        object.__setattr__(result, "_masses", values)
        return result

    @property
    def n_atoms(self) -> int:
        """Number of primitive motif atoms, independent of any supercell."""
        return len(self.atomic_numbers)

    @property
    def cartesian_positions(self) -> np.ndarray:
        """New (n_atoms, 3) Cartesian position array in angstrom, using row lattice vectors."""
        return self.scaled_positions @ self.cell

    def __post_init__(self) -> None:
        """Check ordered blocks, contiguous slices, valid orbit actions and int64 parameter offsets."""
        orders = tuple(block.order for block in self.blocks)
        if not orders or orders != tuple(sorted(set(orders))):
            raise ValueError("cluster-space orders must be nonempty, unique and ascending")
        orbit_start = parameter_start = 0
        for block in self.blocks:
            if block.orbits.start != orbit_start or block.parameters.start != parameter_start:
                raise ValueError(
                    "cluster-space blocks must have contiguous orbit and parameter slices"
                )
            if not orbit_start <= block.orbits.stop <= len(self.orbits):
                raise ValueError("cluster-space orbit slice is outside the model")
            for orbit in self.orbits[block.orbits]:
                if orbit.representative.order != block.order:
                    raise ValueError("orbit order differs from its block")
                if np.any(orbit.operations < 0) or np.any(orbit.operations >= self.symmetry.size):
                    raise ValueError("orbit symmetry operation is outside the group")
                if any(
                    site.site >= self.n_atoms
                    for cluster in orbit.clusters
                    for site in cluster.sites
                ):
                    raise ValueError("orbit site is outside the primitive motif")
            parameter_start += sum(orbit.dimension for orbit in self.orbits[block.orbits])
            require_bound("parameter offsets", parameter_start)
            if block.parameters.stop != parameter_start:
                raise ValueError("parameter slice differs from orbit dimensions")
            orbit_start = block.orbits.stop
        if orbit_start != len(self.orbits):
            raise ValueError("every orbit must belong to one block")

    @property
    def orders(self) -> tuple[int, ...]:
        """Included tensor orders in ascending block order."""
        return tuple(block.order for block in self.blocks)

    def block(self, order: int) -> OrderBlock:
        """Return the slices belonging to one tensor order."""
        for block in self.blocks:
            if block.order == order:
                return block
        raise KeyError(f"cluster space does not contain order {order}")

    @property
    def n_parameters(self) -> int:
        """Total number of physical Cartesian component parameters across all orders."""
        return int(self.blocks[-1].parameters.stop)

    def parameter_name(self, parameter: int) -> str:
        """Return the order, orbit, and local component address of one parameter.

        ``parameter`` is a zero-based index in the global packed parameter
        vector, in ``[0, n_parameters)``. Return a stable descriptive string;
        raise IndexError when the index is outside the cluster-space layout.
        """
        parameter = int(parameter)
        if not 0 <= parameter < self.n_parameters:
            raise IndexError("parameter is outside the cluster space")
        offsets = self.parameter_offsets
        orbit_index = int(np.searchsorted(offsets, parameter, side="right") - 1)
        component = parameter - int(offsets[orbit_index])
        for block in self.blocks:
            if block.parameters.start <= parameter < block.parameters.stop:
                return (
                    f"FC{block.order} orbit {orbit_index - block.orbits.start} "
                    f"component {component}"
                )
        raise RuntimeError("cluster-space parameter layout is inconsistent")

    @property
    def parameter_offsets(self) -> np.ndarray:
        """New readonly int64 array of cumulative orbit dimensions, including the final endpoint."""
        offsets = np.empty(len(self.orbits) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([orbit.dimension for orbit in self.orbits], out=offsets[1:])
        offsets.setflags(write=False)
        return offsets


__all__ = ["ClusterSpace"]
