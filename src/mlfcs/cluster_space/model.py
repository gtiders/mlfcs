"""Immutable cluster-space, cluster, orbit, and order models."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from mlfcs.cluster_space.acoustic import prepare_acoustic_coordinates
from mlfcs.foundation.arrays import as_int64_array, require_bound
from mlfcs.foundation.log import get_logger
from mlfcs.foundation.tensors import tensor_dimension
from mlfcs.geometry.primitive import LatticeSite, primitive_data
from mlfcs.geometry.symmetry import PrimitiveSymmetry


@dataclass(frozen=True, order=True, slots=True)
class Cluster:
    """Ordered collection of lattice sites defining a force-constant term.

    Each site corresponds to one atomic index of the force-constant tensor,
    so repeated sites are allowed. ``order`` is the force-constant order,
    while ``body_order`` counts the distinct atomic positions involved.

    Clusters related by a common lattice translation are equivalent; the first
    site is therefore used as the translational reference.
    """

    sites: tuple[LatticeSite, ...]

    def __post_init__(self) -> None:
        """Express all lattice translations relative to the first site."""
        if len(self.sites) < 2:
            raise ValueError("a cluster must contain at least two sites")

        origin = self.sites[0].translation
        anchored = tuple(
            LatticeSite(
                site.site,
                tuple(value - zero for value, zero in zip(site.translation, origin, strict=True)),
            )
            for site in self.sites
        )
        object.__setattr__(self, "sites", anchored)

    @property
    def order(self) -> int:
        """Force-constant order."""
        return len(self.sites)

    @property
    def body_order(self) -> int:
        """Number of distinct atomic positions in the interaction."""
        return len(set(self.sites))

    @property
    def labels(self) -> tuple[tuple[int, int, int, int], ...]:
        """Lattice-site labels ``(site, tx, ty, tz)`` in tensor index order."""
        return tuple((site.site, *site.translation) for site in self.sites)

    @classmethod
    def from_labels(cls, labels: object) -> Cluster:
        """Construct a cluster from ``(site, tx, ty, tz)`` lattice-site labels."""
        rows = tuple(tuple(int(value) for value in row) for row in labels)

        if any(len(row) != 4 for row in rows):
            raise ValueError("cluster labels must have shape (order, 4)")

        return cls(tuple(LatticeSite(row[0], row[1:]) for row in rows))


@dataclass(frozen=True, slots=True)
class Orbit:
    """Symmetry orbit of a force-constant cluster.

    ``representative`` defines the reference cluster. ``clusters`` contains
    all symmetry-equivalent images generated from it, together with the
    corresponding symmetry operations and tensor-index permutations.

    ``component_basis`` spans the symmetry-allowed Cartesian tensor space.
    Its columns define the independent force-constant parameters, named by
    selected Cartesian tensor components in ``observation_rows``.
    """

    representative: Cluster
    lattice_basis: np.ndarray
    component_basis: np.ndarray
    observation_rows: np.ndarray
    observation_condition: float
    clusters: tuple[Cluster, ...]
    operations: np.ndarray
    permutations: np.ndarray

    def __post_init__(self) -> None:
        """Validate the symmetry orbit and freeze its array data."""
        lattice_basis = as_int64_array(
            self.lattice_basis,
            name="orbit lattice basis",
        )
        component_basis = np.array(
            self.component_basis,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        observation_rows = as_int64_array(
            self.observation_rows,
            name="observation rows",
        )
        operations = as_int64_array(
            self.operations,
            name="orbit operations",
        )
        permutations = as_int64_array(
            self.permutations,
            name="orbit permutations",
        )

        if lattice_basis.ndim != 2 or component_basis.ndim != 2:
            raise ValueError("orbit bases must be matrices")

        if lattice_basis.shape != component_basis.shape:
            raise ValueError("lattice and Cartesian orbit bases have inconsistent shapes")

        if observation_rows.shape != (component_basis.shape[1],):
            raise ValueError("observation rows must select one row per parameter")

        if operations.shape != (len(self.clusters),):
            raise ValueError("orbit operations must have one entry per cluster")

        if permutations.shape != (
            len(self.clusters),
            self.representative.order,
        ):
            raise ValueError("orbit permutations have an inconsistent shape")

        if lattice_basis.shape[0] != tensor_dimension(self.representative.order):
            raise ValueError("orbit basis does not have the declared Cartesian tensor dimension")

        if np.any(observation_rows < 0) or np.any(observation_rows >= component_basis.shape[0]):
            raise ValueError("observation row is outside the tensor")

        if len(np.unique(observation_rows)) != len(observation_rows):
            raise ValueError("observation rows must be distinct")

        if not np.all(np.isfinite(component_basis)):
            raise ValueError("Cartesian orbit basis must be finite")

        expected = np.arange(self.representative.order)
        if not np.array_equal(
            np.sort(permutations, axis=1),
            np.broadcast_to(expected, permutations.shape),
        ):
            raise ValueError("orbit axis actions must be permutations")

        for values in (
            lattice_basis,
            component_basis,
            observation_rows,
            operations,
            permutations,
        ):
            values.setflags(write=False)

        object.__setattr__(self, "lattice_basis", lattice_basis)
        object.__setattr__(self, "component_basis", component_basis)
        object.__setattr__(self, "observation_rows", observation_rows)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "permutations", permutations)

    @property
    def dimension(self) -> int:
        """Number of independent force-constant parameters in the orbit."""
        return int(self.component_basis.shape[1])

    @property
    def observation_matrix(self) -> np.ndarray:
        """Basis rows defining the physical parameter coordinates."""
        return self.component_basis[self.observation_rows]


@dataclass(frozen=True, slots=True)
class OrderBlock:
    """Truncation settings and index ranges for one force-constant order.

    ``order`` specifies the force-constant order, ``cutoff`` limits the
    interaction range, and ``max_body_order`` limits the number of distinct
    atomic positions involved.

    ``orbits`` and ``parameters`` give the corresponding half-open slices in
    the enclosing ``ClusterSpace``.
    """

    order: int
    cutoff: float
    max_body_order: int
    orbits: slice
    parameters: slice

    def __post_init__(self) -> None:
        """Validate and normalize the truncation settings."""
        order = operator.index(self.order)
        body = operator.index(self.max_body_order)
        cutoff = float(self.cutoff)

        if order < 2 or not 1 <= body <= order:
            raise ValueError("order must be >= 2 and max_body_order must lie in 1..order")

        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be a positive finite distance")

        object.__setattr__(self, "order", order)
        object.__setattr__(self, "max_body_order", body)
        object.__setattr__(self, "cutoff", cutoff)


logger = get_logger(__name__)


@dataclass(frozen=True, slots=True, init=False)
class ClusterSpace:
    """Symmetry-reduced force-constant model space built from a periodic reference cell.

    The primitive structure and truncation rules define the candidate
    force-constant clusters. Crystal symmetry groups equivalent clusters into
    ``Orbit`` objects, whose independent Cartesian components form the global
    force-constant parameter space.

    Each force-constant order is described by an ``OrderBlock`` specifying
    its cutoff, maximum body order, and ranges within the global orbit and
    parameter lists.

    Parameters
    ----------
    primitive_atoms : ase.Atoms
        Fully periodic reference structure, which need not be a minimal primitive
        cell. Its atom order and cell basis define the model lattice and are
        retained; geometry and masses are captured independently of the input.
    cutoffs : mapping of int to float
        Pairwise distance cutoff for each included force-constant order, in
        angstrom. Every pair distance in a retained cluster must be strictly
        below the corresponding cutoff.
    max_body_orders : mapping of int to int, optional
        Maximum number of distinct atomic positions allowed at each order.
        Must cover the same orders as ``cutoffs``; omitted values default to
        the corresponding force-constant order.
    symprec : float, default 1e-5
        Positive Cartesian symmetry matching tolerance, in angstrom.

    asr : bool, default False
        Prepare exact sparse acoustic coordinates for fitting and reconstruction.
        Orbit bases and stored physical coefficients remain canonical. Preparation
        may have substantial sparse fill and raises on unsupported integer ranges.

    Notes
    -----
    Initialization discovers symmetry, enumerates clusters and builds the
    invariant bases; this can be expensive. No supercell or training
    structures are owned; ``ClusterMap`` supplies each supercell relation.

    Raises
    ------
    ValueError
        Primitive geometry, truncation inputs or symmetry matching are invalid.
    OverflowError
        A local integer or array-size contract cannot be satisfied.
    """

    cell: np.ndarray
    scaled_positions: np.ndarray
    atomic_numbers: np.ndarray
    symprec: float
    _masses: np.ndarray
    symmetry: PrimitiveSymmetry
    blocks: tuple[OrderBlock, ...]
    orbits: tuple[Orbit, ...]
    _acoustic_coordinates: tuple | None

    def __init__(
        self, primitive_atoms: Atoms, *, cutoffs, max_body_orders=None, symprec=1e-5, asr=False
    ):
        """Construct the symmetry-reduced force-constant space."""
        from mlfcs.cluster_space.construction import construct_space

        if not isinstance(asr, (bool, np.bool_)):
            raise TypeError("asr must be boolean")
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
        coordinates = None
        if asr:
            coordinates = prepare_acoustic_coordinates(self)
        object.__setattr__(self, "_acoustic_coordinates", coordinates)

    @property
    def primitive_atoms(self) -> Atoms:
        """Detached ASE representation of the supplied model reference cell."""
        return Atoms(
            numbers=self.atomic_numbers,
            scaled_positions=self.scaled_positions,
            cell=self.cell,
            masses=self._masses,
            pbc=True,
        )

    @property
    def masses(self) -> np.ndarray:
        """Readonly primitive-site masses, in atomic mass units."""
        return self._masses

    def with_masses(self, masses: object) -> ClusterSpace:
        """Return an equivalent cluster space with new atomic masses.

        Geometry, symmetry, orbits and the parameter layout are preserved.
        ``masses`` must be finite, strictly positive and have shape
        (n_atoms,); invalid values raise ValueError. Rebind existing
        coefficients with ``ForceConstants(new_space, model.coefficients)``.
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
            "_acoustic_coordinates",
        ):
            object.__setattr__(result, name, getattr(self, name))
        object.__setattr__(result, "_masses", values)
        return result

    @property
    def asr(self) -> bool:
        """Whether fitting coordinates enforce translational invariance by construction."""
        return self._acoustic_coordinates is not None

    @property
    def n_free_parameters(self) -> int:
        """Fitting-coordinate dimension; canonical orbit parameter counts remain unchanged."""
        if self._acoustic_coordinates is None:
            return self.n_parameters
        return sum(coordinates.dimension for coordinates in self._acoustic_coordinates)

    def acoustic_coordinates(self, order):
        """Return prepared acoustic coordinates for one order, or None when disabled."""
        if self._acoustic_coordinates is None:
            return None
        return self._acoustic_coordinates[self.orders.index(order)]

    @property
    def n_atoms(self) -> int:
        """Number of primitive motif atoms, independent of any supercell."""
        return len(self.atomic_numbers)

    @property
    def cartesian_positions(self) -> np.ndarray:
        """New (n_atoms, 3) Cartesian position array in angstrom."""
        return self.scaled_positions @ self.cell

    def __post_init__(self) -> None:
        """Validate the consistency of the assembled force-constant space."""
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
        """Included force-constant orders in ascending order."""
        return tuple(block.order for block in self.blocks)

    def block(self, order: int) -> OrderBlock:
        """Return the order block for a force-constant order.

        Missing orders raise KeyError.
        """
        for block in self.blocks:
            if block.order == order:
                return block
        raise KeyError(f"cluster space does not contain order {order}")

    @property
    def n_parameters(self) -> int:
        """Total number of independent force-constant parameters."""
        return int(self.blocks[-1].parameters.stop)

    def parameter_name(self, parameter: int) -> str:
        """Return a stable label for one global force-constant parameter.

        ``parameter`` is a zero-based index in the global packed parameter
        vector, in ``[0, n_parameters)``. Raise IndexError when the index is
        outside the cluster-space layout.
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
        """Cumulative parameter offsets of the symmetry orbits, including the final endpoint."""
        offsets = np.empty(len(self.orbits) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([orbit.dimension for orbit in self.orbits], out=offsets[1:])
        offsets.setflags(write=False)
        return offsets


__all__ = ["Cluster", "ClusterSpace", "Orbit", "OrderBlock"]
