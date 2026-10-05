"""Immutable cluster, orbit, and order records."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np

from mlfcs._arrays import as_int64_array
from mlfcs.geometry.primitive import LatticeSite
from mlfcs.tensors import tensor_dimension


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

        Each row is (primitive_site, tx, ty, tz). Each entry is converted with int. The resulting cluster retains slot order
        and subtracts the first translation from all slots.
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
    lattice_basis : ndarray of int64, shape (3**p, d)
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
    All stored arrays are readonly. Lattice generator orientation does not define
    the public parameter coordinates; each parameter names an observation-row
    Cartesian tensor component.
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
        """Validate tensor/action layouts and store readonly lattice and Cartesian basis arrays."""
        lattice_basis = as_int64_array(self.lattice_basis, name="orbit lattice basis")
        component_basis = np.array(self.component_basis, dtype=np.float64, copy=True, order="C")
        observation_rows = as_int64_array(self.observation_rows, name="observation rows")
        operations = as_int64_array(self.operations, name="orbit operations")
        permutations = as_int64_array(self.permutations, name="orbit permutations")
        if lattice_basis.ndim != 2 or component_basis.ndim != 2:
            raise ValueError("orbit bases must be matrices")
        if lattice_basis.shape != component_basis.shape:
            raise ValueError("lattice and Cartesian orbit bases have inconsistent shapes")
        if observation_rows.shape != (component_basis.shape[1],):
            raise ValueError("observation rows must select one row per parameter")
        if operations.shape != (len(self.clusters),):
            raise ValueError("orbit operations must have one entry per cluster")
        if permutations.shape != (len(self.clusters), self.representative.order):
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
            np.sort(permutations, axis=1), np.broadcast_to(expected, permutations.shape)
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
        """Number of independent Cartesian component parameters in this orbit."""
        return int(self.component_basis.shape[1])

    @property
    def observation_matrix(self) -> np.ndarray:
        """Selected Cartesian basis rows as an advanced-indexed copy, nominally the identity."""
        return self.component_basis[self.observation_rows]


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


__all__ = ["Cluster", "Orbit", "OrderBlock"]
