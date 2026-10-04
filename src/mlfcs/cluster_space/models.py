"""Immutable primitive-cell cluster-space models."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import operator
from dataclasses import asdict, dataclass

import numpy as np
from ase import Atoms

from mlfcs._arrays import integer_array, readonly, require_bound
from mlfcs.core import LatticeSite, PrimitiveSymmetry
from mlfcs.core.log import get_logger
from mlfcs.core.structure import primitive_data, validate_primitive_arrays
from mlfcs.core.tensors import tensor_dimension

logger = get_logger(__name__)


@dataclass(frozen=True, order=True, slots=True)
class Cluster:
    """Immutable ordered tensor slots addressed by primitive lattice sites.

    sites contains at least two LatticeSite values. Construction subtracts the
    first translation from every slot, making its anchor shift zero, but does
    not sort slots. Repeated sites are allowed: order counts slots while
    body_order counts distinct (motif site, translation) addresses.
    """

    sites: tuple[LatticeSite, ...]

    def __post_init__(self) -> None:
        """Require at least two slots and subtract the first lattice translation from every slot."""
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
        """Tensor rank, counting repeated lattice sites as separate slots."""
        return len(self.sites)

    @property
    def body_order(self) -> int:
        """Number of distinct lattice sites, including periodic translations."""
        return len(set(self.sites))

    @property
    def labels(self) -> tuple[tuple[int, int, int, int], ...]:
        """Anchored labels (site, tx, ty, tz), in tensor-slot order."""
        return tuple((site.site, *site.translation) for site in self.sites)

    @classmethod
    def from_labels(cls, labels: object) -> Cluster:
        """Construct and re-anchor a cluster from an (order, 4) label sequence.

        Each row is (primitive_site, tx, ty, tz). Entries are converted with int;
        this convenience constructor does not enforce the strict ndarray integer ABI.
        """
        rows = tuple(tuple(int(value) for value in row) for row in labels)
        if any(len(row) != 4 for row in rows):
            raise ValueError("cluster labels must have shape (order, 4)")
        return cls(tuple(LatticeSite(row[0], row[1:]) for row in rows))


@dataclass(frozen=True, slots=True)
class Orbit:
    """Immutable symmetry orbit with physically named Cartesian parameters.

    Attributes
    ----------
    representative : Cluster
        Anchored cluster used to define independent tensor components.
    exact_lattice_basis : ndarray of int64, shape (3**p, d)
        Saturated invariant lattice-tensor generators, flattened in C order.
    component_basis : ndarray of float64, shape (3**p, d)
        Cartesian parameter-to-component map, with selected observation rows
        equal to identity up to floating roundoff.
    observation_rows : ndarray of int64, shape (d,)
        Representative component indices defining the physical parameter names.
    observation_condition : float
        Condition number of the selected orthonormal observation block.
    clusters : tuple of Cluster
        Distinct images in stored action order.
    operations : ndarray of int64, shape (n_images,)
        Primitive symmetry operation for each image.
    permutations : ndarray of int64, shape (n_images, p)
        Corresponding tensor-axis permutation, using NumPy transpose convention.

    Notes
    -----
    All stored arrays are readonly. Exact generator orientation does not define
    the public parameter coordinates; each parameter names an observation-row
    Cartesian tensor component.
    """

    representative: Cluster
    exact_lattice_basis: np.ndarray
    component_basis: np.ndarray
    observation_rows: np.ndarray
    observation_condition: float
    clusters: tuple[Cluster, ...]
    operations: np.ndarray
    permutations: np.ndarray

    def __post_init__(self) -> None:
        """Validate tensor/action layouts and store readonly exact and Cartesian basis arrays."""
        exact_lattice_basis = integer_array(self.exact_lattice_basis, name="exact orbit basis")
        component_basis = np.array(self.component_basis, dtype=np.float64, copy=True, order="C")
        observation_rows = integer_array(self.observation_rows, name="observation rows")
        operations = integer_array(self.operations, name="orbit operations")
        permutations = integer_array(self.permutations, name="orbit permutations")
        if exact_lattice_basis.ndim != 2 or component_basis.ndim != 2:
            raise ValueError("orbit bases must be matrices")
        if exact_lattice_basis.shape != component_basis.shape:
            raise ValueError("exact and Cartesian orbit bases have inconsistent shapes")
        if observation_rows.shape != (component_basis.shape[1],):
            raise ValueError("observation rows must select one row per parameter")
        if operations.shape != (len(self.clusters),):
            raise ValueError("orbit operations must have one entry per cluster")
        if permutations.shape != (len(self.clusters), self.representative.order):
            raise ValueError("orbit permutations have an inconsistent shape")
        if exact_lattice_basis.shape[0] != tensor_dimension(self.representative.order):
            raise ValueError("orbit basis does not have the declared Cartesian tensor dimension")
        if np.any(observation_rows < 0) or np.any(observation_rows >= component_basis.shape[0]):
            raise ValueError("observation row is outside the tensor")
        if len(np.unique(observation_rows)) != len(observation_rows):
            raise ValueError("observation rows must be distinct")
        if not np.all(np.isfinite(component_basis)):
            raise ValueError("Cartesian orbit basis must be finite")
        expected = np.arange(self.representative.order)
        if not np.array_equal(
            np.sort(permutations, axis=1), np.broadcast_to(expected, permutations.shape)
        ):
            raise ValueError("orbit axis actions must be permutations")
        for values in (
            exact_lattice_basis,
            component_basis,
            observation_rows,
            operations,
            permutations,
        ):
            values.setflags(write=False)
        object.__setattr__(self, "exact_lattice_basis", exact_lattice_basis)
        object.__setattr__(self, "component_basis", component_basis)
        object.__setattr__(self, "observation_rows", observation_rows)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "permutations", permutations)

    @property
    def dimension(self) -> int:
        """Number of independent Cartesian component parameters in this orbit."""
        return int(self.component_basis.shape[1])

    @property
    def observation_matrix(self) -> np.ndarray:
        """Selected Cartesian basis rows as an advanced-indexed copy, nominally the identity."""
        return self.component_basis[self.observation_rows]

    def __reduce__(self):
        """Serialize orbit geometry and bases for validated dataclass reconstruction."""
        return type(self), (
            self.representative,
            self.exact_lattice_basis,
            self.component_basis,
            self.observation_rows,
            self.observation_condition,
            self.clusters,
            self.operations,
            self.permutations,
        )


@dataclass(frozen=True, slots=True)
class OrderBlock:
    """Immutable truncation inputs and contiguous slices for one tensor order.

    order is the tensor rank, cutoff a positive distance in angstrom, and
    max_body_order limits distinct lattice addresses, not tensor slots.
    orbits and parameters are half-open global slices into ClusterSpace;
    the enclosing model validates their contiguity and dimensions.
    """

    order: int
    cutoff: float
    max_body_order: int
    orbits: slice
    parameters: slice

    def __post_init__(self) -> None:
        """Normalize truncation inputs and require positive cutoff and a valid body order."""
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
        from mlfcs.cluster_space.builder import _construct_space

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
        symmetry, blocks, orbits = _construct_space(
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Cluster-space identity: fingerprint=%s", self.fingerprint)

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
        The parameter-space fingerprint is unchanged. Rebind existing physical
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

    def __reduce__(self):
        """Serialize model state without requiring orbit enumeration during restoration."""
        return _restore_cluster_space, (self._state(),)

    def _state(self):
        """Collect geometry, symmetry, truncation and basis data for validated restoration."""
        return {
            "cell": self.cell,
            "scaled_positions": self.scaled_positions,
            "atomic_numbers": self.atomic_numbers,
            "symprec": self.symprec,
            "masses": self._masses,
            "symmetry": asdict(self.symmetry),
            "blocks": [asdict(b) for b in self.blocks],
            "orbits": [
                {
                    "representative": o.representative.labels,
                    "exact_lattice_basis": o.exact_lattice_basis,
                    "component_basis": o.component_basis,
                    "observation_rows": o.observation_rows,
                    "observation_condition": o.observation_condition,
                    "clusters": [c.labels for c in o.clusters],
                    "operations": o.operations,
                    "permutations": o.permutations,
                }
                for o in self.orbits
            ],
        }

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

    @property
    def parameter_offsets(self) -> np.ndarray:
        """New readonly int64 array of cumulative orbit dimensions, including the final endpoint."""
        offsets = np.empty(len(self.orbits) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([orbit.dimension for orbit in self.orbits], out=offsets[1:])
        offsets.setflags(write=False)
        return offsets

    @property
    def fingerprint(self) -> str:
        """Stable physical identity independent of exact kernel generator columns."""

        def float_rows(values: np.ndarray) -> list[list[str]]:
            """Encode floating-point rows exactly as hexadecimal strings for stable hashing."""
            return [[float(value).hex() for value in row] for row in values]

        payload = {
            "schema": 1,
            "cell": float_rows(self.cell),
            "scaled_positions": float_rows(self.scaled_positions),
            "numbers": [int(value) for value in self.atomic_numbers],
            "symprec": self.symprec.hex(),
            "blocks": [
                {
                    "order": block.order,
                    "cutoff": block.cutoff.hex(),
                    "max_body_order": block.max_body_order,
                    "orbit_slice": [block.orbits.start, block.orbits.stop],
                    "parameter_slice": [block.parameters.start, block.parameters.stop],
                }
                for block in self.blocks
            ],
            "rotations": self.symmetry.rotations.tolist(),
            "site_permutations": self.symmetry.site_permutations.tolist(),
            "site_shifts": self.symmetry.site_shifts.tolist(),
            "orbits": [
                {
                    "representative": orbit.representative.labels,
                    "dimension": orbit.dimension,
                    "observation_rows": orbit.observation_rows.tolist(),
                }
                for orbit in self.orbits
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["Cluster", "ClusterSpace", "Orbit", "OrderBlock"]


def _restore_cluster_space(state):
    """Validate serialized model data without repeating orbit construction."""
    cell, positions, numbers, symprec = validate_primitive_arrays(
        state["cell"], state["scaled_positions"], state["atomic_numbers"], state["symprec"]
    )
    symmetry = PrimitiveSymmetry(**state["symmetry"])
    # Spaces serialized by earlier versions may carry a per-order bounds block.
    blocks = tuple(
        OrderBlock(**{k: v for k, v in b.items() if k != "bounds"}) for b in state["blocks"]
    )
    orbits = tuple(
        Orbit(
            **{
                **o,
                "representative": Cluster.from_labels(o["representative"]),
                "clusters": tuple(Cluster.from_labels(c) for c in o["clusters"]),
            }
        )
        for o in state["orbits"]
    )
    if symmetry.symprec != symprec or symmetry.site_permutations.shape[1] != len(numbers):
        raise ValueError("serialized symmetry acts on a different primitive structure")
    masses = readonly(state.get("masses", Atoms(numbers=numbers).get_masses()), np.float64)
    if masses.shape != numbers.shape or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("serialized atomic masses must be positive and finite")
    result = object.__new__(ClusterSpace)
    for name, value in (
        ("cell", cell),
        ("scaled_positions", positions),
        ("atomic_numbers", numbers),
        ("symprec", symprec),
        ("_masses", masses),
        ("symmetry", symmetry),
        ("blocks", blocks),
        ("orbits", orbits),
    ):
        object.__setattr__(result, name, value)
    result.__post_init__()
    return result
