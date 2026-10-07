"""Primitive-first force-constant tensors, periodic folding and blockwise composition."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from types import MappingProxyType

import numpy as np
from numba import njit

from mlfcs.force_constants.expansion import image_tensor
from mlfcs.force_constants.model import ForceConstants
from mlfcs.foundation.arrays import as_int64_array, readonly, require_allocation
from mlfcs.foundation.tensors import tensor_dimension
from mlfcs.geometry.primitive import LatticeSite
from mlfcs.mapping import ClusterMap

DEFAULT_BLOCK_BYTES = 8 * 1024**2


def _block_capacity(order, max_bytes):
    """Reserve room for output, a consumer copy and tensor-action temporaries."""
    budget = operator.index(max_bytes)
    tensor_bytes = 8 * tensor_dimension(order)
    label_bytes = 8 * (4 * order - 3)
    capacity = (budget - (order + 3) * tensor_bytes) // (2 * tensor_bytes + label_bytes)
    if capacity < 1:
        raise ValueError("max_bytes cannot hold one tensor and its working arrays")
    return capacity


def _tiles(shape, capacity):
    """Partition atom axes into Cartesian-complete blocks of bounded tensor count."""
    widths = [1] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        widths[axis] = min(shape[axis], capacity)
        capacity = max(1, capacity // widths[axis])
    for starts in product(*(range(0, size, width) for size, width in zip(shape, widths))):
        yield tuple(
            slice(start, min(start + width, size))
            for start, width, size in zip(starts, widths, shape)
        )


def _selected_entries(indices, axes, positions, depth=0, local=()):
    """Traverse only image-index branches intersecting the requested atom block."""
    if depth == len(axes):
        yield local, indices
        return
    if len(indices) < len(axes[depth]):
        for atom, child in indices.items():
            if atom in positions[depth]:
                yield from _selected_entries(
                    child, axes, positions, depth + 1, (*local, positions[depth][atom])
                )
    else:
        for index, atom in enumerate(axes[depth]):
            if int(atom) in indices:
                yield from _selected_entries(
                    indices[int(atom)], axes, positions, depth + 1, (*local, index)
                )


@njit(cache=True)
def _accumulate_forces(result, tensors, displacements, first, second):
    """Contract one full-order FC2 block with Cartesian displacement frames."""
    for frame in range(len(displacements)):
        for i in range(tensors.shape[0]):
            for j in range(tensors.shape[1]):
                for a in range(3):
                    for b in range(3):
                        result[frame, first + i, a] -= (
                            tensors[i, j, a, b] * displacements[frame, second + j, b]
                        )


@dataclass(frozen=True, slots=True, init=False)
class CompactForceConstants:
    """Force-constant tensors anchored on primitive sites of a periodic supercell.

    ``source`` is a ForceConstants model or a mapping from tensor order to
    already folded arrays. For order p, folded arrays have atom axes
    (N_primitive, N_supercell, ..., N_supercell), followed by p Cartesian axes.
    Their first atom is the zero-translation primitive representative, not
    necessarily the first corresponding atom in the input supercell.

    Values are energy derivatives in eV/angstrom**p, without mass weighting
    or Taylor factorials. Arrays must use the supplied ClusterMap's atom order
    and periodic boundary. Callers are responsible for matching physical
    geometry and conventions; shapes and finite values are checked.

    Parameter sources retain individual lattice images. Folded sources do not
    contain that information and cannot supply lattice_blocks for their orders.
    Addition combines declared contributions without projecting onto a basis;
    it does not recover lattice information lost by folding. Missing orders
    raise ValueError. Arrays are only materialized by explicit array methods.

    A map may be omitted for a parameter source used solely for lattice export.
    No Ewald object, physical evaluator or serialization is stored here.
    """

    cluster_map: ClusterMap | None
    _sources: tuple
    _indices: dict

    def __init__(self, source, cluster_map=None):
        """Capture a parameter source or normalized folded tensors."""
        if cluster_map is not None and not isinstance(cluster_map, ClusterMap):
            raise TypeError("cluster_map must be a ClusterMap")
        if isinstance(source, ForceConstants):
            prepared = source
        elif isinstance(source, Mapping):
            if cluster_map is None:
                raise ValueError("folded tensors require a ClusterMap")
            arrays = {}
            for key, values in source.items():
                order = operator.index(key)
                if order < 2:
                    raise ValueError("force-constant order must be at least two")
                shape = (
                    (cluster_map.cluster_space.n_atoms,)
                    + (cluster_map.n_atoms,) * (order - 1)
                    + (3,) * order
                )
                require_allocation("compact force constants", shape)
                array = readonly(values, np.float64)
                if array.shape != shape or not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"order-{order} compact tensors must be finite with shape {shape}"
                    )
                arrays[order] = array
            if not arrays:
                raise ValueError("force constants must contain at least one order")
            prepared = MappingProxyType(arrays)
        else:
            raise TypeError("source must be ForceConstants or a mapping of compact tensors")
        object.__setattr__(self, "cluster_map", cluster_map)
        object.__setattr__(self, "_sources", (prepared,))
        object.__setattr__(self, "_indices", {})

    @property
    def cluster_space(self):
        """Primitive geometry used to interpret tensor labels."""
        if self.cluster_map is not None:
            return self.cluster_map.cluster_space
        return self._sources[0].cluster_space

    @property
    def orders(self):
        """Tensor orders explicitly supplied by at least one contribution."""
        return tuple(sorted({order for source in self._sources for order in self._orders(source)}))

    @staticmethod
    def _orders(source):
        """Read declared orders without treating unspecified ones as zero models."""
        return source.orders if isinstance(source, ForceConstants) else tuple(source)

    def __add__(self, other):
        """Combine tensor contributions; callers must pair matching maps and units."""
        if not isinstance(other, CompactForceConstants):
            return NotImplemented
        if (self.cluster_map is None) != (other.cluster_map is None):
            raise ValueError("both tensor representations must have a map, or neither")
        if self.cluster_map is not None and (
            self.cluster_map.n_atoms != other.cluster_map.n_atoms
            or self.cluster_space.n_atoms != other.cluster_space.n_atoms
        ):
            raise ValueError("compact tensor dimensions differ")
        result = object.__new__(type(self))
        object.__setattr__(result, "cluster_map", self.cluster_map)
        object.__setattr__(result, "_sources", self._sources + other._sources)
        object.__setattr__(result, "_indices", {})
        return result

    def _require_order(self, order):
        """Require a declared tensor order before evaluating any contribution."""
        order = operator.index(order)
        if order not in self.orders:
            raise ValueError(f"force constants do not contain order {order}")
        return order

    def _lattice_index(self, source_index, order):
        """Group orbit-image references by unambiguous primitive lattice labels."""
        key = ("lattice", source_index, order)
        if key not in self._indices:
            model = self._sources[source_index]
            indices = {}
            offset = 0
            for orbit in model.cluster_space.orbits[model.cluster_space.block(order).orbits]:
                stop = offset + orbit.dimension
                for image, cluster in enumerate(orbit.clusters):
                    label = (
                        tuple(site.site for site in cluster.sites),
                        tuple(site.translation for site in cluster.sites[1:]),
                    )
                    if label in indices:
                        raise RuntimeError(
                            "cluster-space expansion produced a duplicate lattice cluster"
                        )
                    indices[label] = (orbit, model.coefficients[order][offset:stop], image)
                offset = stop
            if offset != len(model.coefficients[order]):
                raise RuntimeError(
                    "orbit traversal did not consume all force-constant coefficients"
                )
            self._indices[key] = indices
        return self._indices[key]

    def _folded_index(self, source_index, order):
        """Index image aliases once so blocks never scan unrelated orbit images."""
        key = ("folded", source_index, order)
        if key not in self._indices:
            indices = {}
            for (sites, translations), entry in self._lattice_index(source_index, order).items():
                atoms = (sites[0],) + tuple(
                    self.cluster_map.atom_index(LatticeSite(site, translation))
                    for site, translation in zip(sites[1:], translations)
                )
                branch = indices
                for atom in atoms[:-1]:
                    branch = branch.setdefault(atom, {})
                branch.setdefault(atoms[-1], []).append(entry)
            self._indices[key] = indices
        return self._indices[key]

    def _compact_block(self, order, axes):
        """Evaluate and add the contributions at requested compact atom indices."""
        shape = tuple(len(axis) for axis in axes) + (3,) * order
        require_allocation("tensor block", shape)
        result = np.zeros(shape)
        for source_index, source in enumerate(self._sources):
            if order not in self._orders(source):
                continue
            if isinstance(source, ForceConstants):
                indices = self._folded_index(source_index, order)
                positions = [{int(atom): i for i, atom in enumerate(axis)} for axis in axes]
                previous_orbit = None
                for local, entries in _selected_entries(indices, axes, positions):
                    for orbit, coefficients, image in entries:
                        if orbit is not previous_orbit:
                            representative = (orbit.component_basis @ coefficients).reshape(
                                (3,) * order
                            )
                            previous_orbit = orbit
                        result[local] += image_tensor(source, orbit, representative, image)
            else:
                result += source[order][np.ix_(*axes)]
        if not np.all(np.isfinite(result)):
            raise ArithmeticError("tensor evaluation produced nonfinite force constants")
        return result

    def _shape(self, order, *, full=False):
        """Require a target supercell and return its requested atom-axis dimensions."""
        if self.cluster_map is None:
            raise ValueError("periodic compact/full tensors require a ClusterMap")
        first = self.cluster_map.n_atoms if full else self.cluster_space.n_atoms
        return (first,) + (self.cluster_map.n_atoms,) * (order - 1)

    def compact_blocks(self, order, *, max_bytes=DEFAULT_BLOCK_BYTES):
        """Yield primitive-first atom slices and complete Cartesian tensor blocks.

        The workspace budget covers tensor output, one consumer copy and
        action temporaries, excluding source arrays and derived label indices.
        Yielded arrays are independent; retaining every block forfeits streaming.
        """
        order = self._require_order(order)
        capacity = _block_capacity(order, max_bytes)
        for slices in _tiles(self._shape(order), capacity):
            axes = tuple(np.arange(axis.start, axis.stop) for axis in slices)
            yield slices, self._compact_block(order, axes)

    def full_blocks(self, order, *, max_bytes=DEFAULT_BLOCK_BYTES):
        """Yield full-supercell blocks by translation, without a full tensor array."""
        order = self._require_order(order)
        capacity = _block_capacity(order, max_bytes)
        shape = self._shape(order, full=True)
        tail_shape = (1,) + shape[1:]
        labels = np.column_stack(
            (self.cluster_map.primitive_site_indices, self.cluster_map.lattice_translations)
        )
        for first in range(shape[0]):
            origin = self.cluster_map.lattice_translations[first]
            translation = as_int64_array(
                [[-int(value) for value in origin]], name="inverse supercell translation"
            )
            translated = self.cluster_map.map_labels(labels, translation)[:, 0]
            for slices in _tiles(tail_shape, capacity):
                axes = (np.array([self.cluster_map.primitive_site_indices[first]]),) + tuple(
                    translated[axis] for axis in slices[1:]
                )
                yield (slice(first, first + 1), *slices[1:]), self._compact_block(order, axes)

    def compact_array(self, order):
        """Materialize primitive-first tensors in eV/angstrom**order."""
        order = self._require_order(order)
        shape = self._shape(order) + (3,) * order
        require_allocation("compact tensor array", shape)
        result = np.empty(shape)
        for slices, tensors in self.compact_blocks(order):
            result[slices] = tensors
            del tensors
        return result

    def full_array(self, order):
        """Explicitly materialize tensors with every atom axis in supercell order."""
        order = self._require_order(order)
        shape = self._shape(order, full=True) + (3,) * order
        require_allocation("full tensor array", shape)
        result = np.empty(shape)
        for slices, tensors in self.full_blocks(order):
            result[slices] = tensors
            del tensors
        return result

    def lattice_blocks(self, order, *, max_bytes=DEFAULT_BLOCK_BYTES):
        """Yield sites, relative translations and unfolded Cartesian image tensors.

        Translations have shape (batch, order-1, 3); the first site's shift is
        zero. Folded sources cannot recover these labels and raise ValueError,
        even when combined with a parameter source carrying the same order.
        """
        order = self._require_order(order)
        if any(
            not isinstance(source, ForceConstants)
            for source in self._sources
            if order in self._orders(source)
        ):
            raise ValueError(f"order-{order} contains folded tensors without lattice-image data")
        capacity = _block_capacity(order, max_bytes)
        grouped = {}
        for source_index, source in enumerate(self._sources):
            if order not in self._orders(source):
                continue
            for label, entry in self._lattice_index(source_index, order).items():
                grouped.setdefault(label, []).append((source, entry))
        labels = sorted(grouped)
        for start in range(0, len(labels), capacity):
            batch = labels[start : start + capacity]
            tensors = np.zeros((len(batch),) + (3,) * order)
            for index, label in enumerate(batch):
                for source, (orbit, coefficients, image) in grouped[label]:
                    representative = (orbit.component_basis @ coefficients).reshape((3,) * order)
                    tensors[index] += image_tensor(source, orbit, representative, image)
            if not np.all(np.isfinite(tensors)):
                raise ArithmeticError("tensor evaluation produced nonfinite force constants")
            yield (
                np.asarray([label[0] for label in batch], dtype=np.int64),
                np.asarray([label[1] for label in batch], dtype=np.int64),
                tensors,
            )
            del tensors

    def harmonic_forces(self, displacements):
        """Return the FC2 force -Phi u, in eV/angstrom, without applying higher orders.

        Displacements are Cartesian angstrom arrays (atoms, 3) or
        (frames, atoms, 3), in mapped supercell order. The result preserves that
        shape. FC2 blocks are contracted without materializing a full Hessian.
        """
        self._require_order(2)
        n = self._shape(2, full=True)[0]
        values = np.asarray(displacements, dtype=np.float64)
        single = values.shape == (n, 3)
        if single:
            values = values[None]
        if values.ndim != 3 or values.shape[1:] != (n, 3) or not np.all(np.isfinite(values)):
            raise ValueError(
                "displacements must be finite with shape (atoms, 3) or (frames, atoms, 3)"
            )
        require_allocation("harmonic forces", values.shape)
        result = np.zeros(values.shape)
        for slices, tensors in self.full_blocks(2):
            _accumulate_forces(result, tensors, values, slices[0].start, slices[1].start)
            del tensors
        if not np.all(np.isfinite(result)):
            raise ArithmeticError("force contraction produced nonfinite values")
        return result[0] if single else result

    def write(self, file, *, format, order, storage=None, threshold=1e-8):
        """Write one order using the shared external-format writer.

        Phonopy FC2 and phono3py FC3 consume periodic blocks. ShengBTE FC3/FC4
        requires individual lattice images. Unsupported
        orders or unavailable lattice data raise ValueError. The threshold,
        in this order's physical units, is applied after contributions sum.
        This object has no native save/load interface.
        """
        from mlfcs.force_constants.formats import write

        return write(self, file, format=format, order=order, storage=storage, threshold=threshold)


__all__ = ["CompactForceConstants"]
