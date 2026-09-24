"""Immutable primitive-cell cluster-space models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from mlfcs.core import LatticeSite, PrimitiveCell, PrimitiveSymmetry


@dataclass(frozen=True, order=True, slots=True)
class Cluster:
    """An anchored tuple of primitive lattice sites."""

    sites: tuple[LatticeSite, ...]

    def __post_init__(self) -> None:
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
        return len(self.sites)

    @property
    def body_order(self) -> int:
        return len(set(self.sites))

    @property
    def labels(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple((site.site, *site.translation) for site in self.sites)

    @classmethod
    def from_labels(cls, labels: object) -> Cluster:
        rows = tuple(tuple(int(value) for value in row) for row in labels)
        if any(len(row) != 4 for row in rows):
            raise ValueError("cluster labels must have shape (order, 4)")
        return cls(tuple(LatticeSite(row[0], row[1:]) for row in rows))


@dataclass(frozen=True, slots=True)
class Orbit:
    """One symmetry orbit with physically named Cartesian parameters.

    ``component_basis`` maps parameters to the representative Cartesian tensor.
    Its selected rows are the identity, so every parameter is literally one
    representative tensor component; it does not inherit an arbitrary Smith
    basis orientation.
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
        exact_lattice_basis = np.array(self.exact_lattice_basis, dtype=object, copy=True)
        component_basis = np.array(self.component_basis, dtype=np.float64, copy=True, order="C")
        observation_rows = np.array(self.observation_rows, dtype=np.int32, copy=True)
        operations = np.array(self.operations, dtype=np.int32, copy=True)
        permutations = np.array(self.permutations, dtype=np.int16, copy=True)
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
        return int(self.component_basis.shape[1])

    @property
    def observation_matrix(self) -> np.ndarray:
        return self.component_basis[self.observation_rows]


@dataclass(frozen=True, slots=True)
class IntBounds:
    """Proven integer bounds for one tensor order."""

    translation: int
    rotation: int
    shift: int
    labels: int
    headroom: int
    tensor: int
    kernel: int


@dataclass(frozen=True, slots=True)
class OrderBlock:
    """Slices and physical truncation inputs for one tensor order."""

    order: int
    cutoff: float
    max_body_order: int
    orbits: slice
    parameters: slice
    bounds: IntBounds


@dataclass(frozen=True, slots=True)
class ClusterSpace:
    """A primitive-cell model space with no supercell or training state."""

    primitive: PrimitiveCell
    symmetry: PrimitiveSymmetry
    blocks: tuple[OrderBlock, ...]
    orbits: tuple[Orbit, ...]

    def __post_init__(self) -> None:
        orders = tuple(block.order for block in self.blocks)
        if not orders or orders != tuple(sorted(set(orders))):
            raise ValueError("cluster-space orders must be nonempty, unique and ascending")

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(block.order for block in self.blocks)

    def block(self, order: int) -> OrderBlock:
        """Return the slices belonging to one tensor order."""
        for block in self.blocks:
            if block.order == order:
                return block
        raise KeyError(f"cluster space does not contain order {order}")

    @property
    def n_parameters(self) -> int:
        return int(self.blocks[-1].parameters.stop)

    @property
    def parameter_offsets(self) -> np.ndarray:
        offsets = np.empty(len(self.orbits) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([orbit.dimension for orbit in self.orbits], out=offsets[1:])
        offsets.setflags(write=False)
        return offsets

    @property
    def fingerprint(self) -> str:
        """Stable identity independent of the particular SNF kernel columns."""

        def float_rows(values: np.ndarray) -> list[list[str]]:
            return [[float(value).hex() for value in row] for row in values]

        payload = {
            "schema": 1,
            "cell": float_rows(self.primitive.cell),
            "scaled_positions": float_rows(self.primitive.scaled_positions),
            "numbers": [int(value) for value in self.primitive.numbers],
            "symprec": self.primitive.symprec.hex(),
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


__all__ = ["Cluster", "ClusterSpace", "IntBounds", "Orbit", "OrderBlock"]
