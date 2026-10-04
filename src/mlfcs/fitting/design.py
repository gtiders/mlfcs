"""Compiled Cartesian force designs and caller-owned workspace."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from time import perf_counter

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

from mlfcs._arrays import integer_array, readonly, require_allocation
from mlfcs.core.log import get_logger
from mlfcs.core.tensors import rotate_basis
from mlfcs.mapping import ClusterMap
from mlfcs.mapping.geometry import mapped_labels

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OrderDesign:
    """Readonly ragged Cartesian design buffers for one tensor order.

    components is (3**p, p) in C-order direction enumeration. Orbit offsets
    index image ranges; parameter starts refer to the global ClusterSpace.
    image_atoms is (images, translations, p); flattened image_basis stores
    (3**p, orbit_dimension) blocks delimited by image_basis_offsets.
    factor is 1/p!, with differentiation over every tensor slot in the kernel.
    """

    factor: float
    components: np.ndarray
    orbit_image_offsets: np.ndarray
    orbit_parameter_starts: np.ndarray
    orbit_dimensions: np.ndarray
    image_atoms: np.ndarray
    image_basis: np.ndarray
    image_basis_offsets: np.ndarray

    def __post_init__(self):
        """Normalize integer metadata and float64 basis storage to readonly contiguous arrays."""
        for name in (
            "components",
            "orbit_image_offsets",
            "orbit_parameter_starts",
            "orbit_dimensions",
            "image_atoms",
            "image_basis_offsets",
        ):
            array = getattr(self, name)
            require_allocation(name, array.shape)
            object.__setattr__(self, name, integer_array(array, name=name))
        require_allocation("image basis", self.image_basis.shape)
        object.__setattr__(self, "image_basis", readonly(self.image_basis, np.float64))

    @property
    def max_dimension(self) -> int:
        """Largest orbit parameter dimension, or zero for an empty order."""
        return int(self.orbit_dimensions.max()) if len(self.orbit_dimensions) else 0


def _compile_order(
    cluster_map: ClusterMap, order: int, translations: tuple[tuple[int, int, int], ...]
) -> OrderDesign:
    """Map translated orbit images and flatten rotated Cartesian bases for force evaluation.

    Translations are primitive integer triples. Preserve orbit/image order and
    global parameter offsets; return readonly OrderDesign buffers. Cartesian
    row-action matrices are transposed for the tensor contraction convention.
    """
    space = cluster_map.cluster_space
    block = space.block(order)
    parameter_offsets = space.parameter_offsets
    orbit_indices = range(block.orbits.start, block.orbits.stop)
    image_count = sum(len(space.orbits[i].clusters) for i in orbit_indices)
    basis_count = sum(
        len(space.orbits[i].clusters) * 3**order * space.orbits[i].dimension for i in orbit_indices
    )
    require_allocation("image atoms", (image_count, len(translations), order))
    require_allocation("image basis", (basis_count,))
    require_allocation("image basis offsets", (image_count + 1,))
    require_allocation("orbit offsets", (len(orbit_indices) + 1,))
    require_allocation("tensor components", (3**order, order))
    translations = integer_array(translations, name="mapping translations")
    orbit_image_offsets = [0]
    parameter_starts = []
    dimensions = []
    image_atoms = []
    basis_values = []
    basis_offsets = [0]
    for orbit_index in orbit_indices:
        orbit = space.orbits[orbit_index]
        dimension = orbit.dimension
        parameter_starts.append(int(parameter_offsets[orbit_index]))
        dimensions.append(dimension)
        for image, cluster in enumerate(orbit.clusters):
            labels = np.asarray(cluster.labels, dtype=np.int64)
            atoms = mapped_labels(labels, translations, cluster_map._periodic).T.copy()
            image_atoms.append(atoms)
            rotation = space.symmetry.cartesian_rotations[orbit.operations[image]].T
            basis = rotate_basis(orbit.component_basis, rotation, orbit.permutations[image])
            flattened = np.ascontiguousarray(basis).reshape(-1)
            basis_values.append(flattened)
            basis_offsets.append(basis_offsets[-1] + len(flattened))
        orbit_image_offsets.append(len(image_atoms))
    return OrderDesign(
        factor=1.0 / factorial(order),
        components=np.ascontiguousarray(
            np.asarray(tuple(np.ndindex((3,) * order)), dtype=np.int64)
        ),
        orbit_image_offsets=np.asarray(orbit_image_offsets, dtype=np.int64),
        orbit_parameter_starts=np.asarray(parameter_starts, dtype=np.int64),
        orbit_dimensions=np.asarray(dimensions, dtype=np.int64),
        image_atoms=(
            np.ascontiguousarray(image_atoms, dtype=np.int64)
            if image_atoms
            else np.empty((0, len(translations), order), dtype=np.int64)
        ),
        image_basis=np.concatenate(basis_values) if basis_values else np.zeros(0),
        image_basis_offsets=np.asarray(basis_offsets, dtype=np.int64),
    )


class ForceDesign:
    """Reusable force-design metadata for every order in one ClusterMap.

    Parameters
    ----------
    cluster_map : ClusterMap
        Supercell relation retained by reference. Its exact folded rank must
        preserve all primitive parameters, otherwise AliasingError is raised.

    Notes
    -----
    Construction compiles translated atom indices and rotated float64 bases.
    Rows are atom-major x/y/z forces; columns follow ClusterSpace parameters.
    Workspace is caller-owned and mutable, separate from readonly design data.
    """

    __slots__ = ("cluster_map", "orders", "rows")

    def __init__(self, cluster_map: ClusterMap):
        """Require full structural rank and compile readonly per-order force-design buffers."""
        started = perf_counter()
        logger.info(
            "Force-design compilation started: supercell_atoms=%d orders=%s parameters=%d",
            cluster_map.n_atoms,
            cluster_map.cluster_space.orders,
            cluster_map.cluster_space.n_parameters,
        )
        cluster_map.rank_info().require_full()
        self.cluster_map = cluster_map
        self.rows = 3 * len(cluster_map.atomic_numbers)
        require_allocation("force design", (self.rows, self.n_parameters))
        translations = cluster_map.translation_representatives
        self.orders = tuple(
            _compile_order(cluster_map, order, translations)
            for order in cluster_map.cluster_space.orders
        )
        for order, compiled in zip(cluster_map.cluster_space.orders, self.orders, strict=True):
            logger.info(
                "Force-design order ready: order=%d images=%d basis_bytes=%d",
                order,
                len(compiled.image_atoms),
                compiled.image_basis.nbytes,
            )
        logger.info(
            "Force-design compilation complete: rows=%d columns=%d elapsed_s=%.2f",
            self.rows,
            self.n_parameters,
            perf_counter() - started,
        )

    @property
    def n_parameters(self) -> int:
        """Total primitive parameter count, including every order in the cluster space."""
        return self.cluster_map.cluster_space.n_parameters

    def allocate_workspace(self):
        """Allocate per-order, per-thread float64 scratch for repeated matrix calls.

        Do not share the returned workspace between concurrent calls. Its thread
        capacity is captured from the current Numba thread count.
        """
        workspace = allocate_design_workspace(
            self.rows, tuple(order.max_dimension for order in self.orders)
        )
        logger.debug(
            "Design workspace allocated: threads=%d scratch_bytes=%d",
            workspace.threads,
            sum(array.nbytes for array in workspace.scratch),
        )
        return workspace

    def matrix(self, displacement: np.ndarray, *, workspace=None) -> np.ndarray:
        """Return the physical force-design matrix for one displacement snapshot.

        Parameters
        ----------
        displacement : array_like
            n_atoms*3 finite Cartesian components in atom-major x/y/z order, in
            angstrom; an (n_atoms, 3) array is accepted by flattening.
        workspace : DesignWorkspace, optional
            Matching mutable scratch, reused in place. Omission allocates scratch.

        Returns
        -------
        matrix : ndarray of float64, shape (3*n_atoms, n_parameters)
            New unnormalized matrix A such that predicted forces are A @ parameters.

        Notes
        -----
        The force sign, Taylor factorial and every translated orbit image are
        included. Displacements are not mutated; workspace must not be shared
        between concurrent calls. Invalid shape, values or scratch raise ValueError.
        """
        values = np.ascontiguousarray(displacement, dtype=np.float64).reshape(-1)
        if values.shape != (self.rows,):
            raise ValueError(f"displacement must contain {self.rows} Cartesian components")
        if not np.all(np.isfinite(values)):
            raise ValueError("displacement must be finite")
        if workspace is None:
            workspace = self.allocate_workspace()
        workspace.validate(self.rows, tuple(order.max_dimension for order in self.orders))
        result = np.zeros((self.rows, self.n_parameters), dtype=np.float64)
        for order, scratch in zip(self.orders, workspace.scratch, strict=True):
            accumulate_design(
                values,
                result,
                order.orbit_image_offsets,
                order.orbit_parameter_starts,
                order.orbit_dimensions,
                order.image_atoms,
                order.image_basis,
                order.image_basis_offsets,
                order.components,
                order.factor,
                scratch,
            )
        return result


@njit(cache=True, parallel=True, nogil=True)
def accumulate_design(
    displacement,
    out,
    orbit_image_offsets,
    orbit_parameter_starts,
    orbit_dimensions,
    image_atoms,
    image_basis,
    image_basis_offsets,
    components,
    factor,
    scratch,
):
    """Accumulate one order into a physical force-design matrix in place.

    Displacement is atom-major Cartesian float64; out has shape (rows, total
    parameters). Compiled ragged offsets delimit images and flattened bases.
    Each tensor slot contributes -1/p! times the product of all other slot
    displacements. scratch has one (rows, max_orbit_dimension) block per thread.
    Orbits write disjoint parameter columns, so prange needs no atomics; each
    thread clears its scratch before use. Validated metadata admits all indices.
    """
    orbit_count = orbit_dimensions.size
    translation_count = image_atoms.shape[1]
    order = image_atoms.shape[2]
    component_count = components.shape[0]
    rows = displacement.size
    # Orbit columns are disjoint even when their images share force rows.
    # Each thread therefore accumulates privately, then writes its own columns.
    for orbit in prange(orbit_count):
        local = scratch[get_thread_id()]
        dimension_count = orbit_dimensions[orbit]
        parameter_start = orbit_parameter_starts[orbit]
        for row in range(rows):
            for dimension in range(dimension_count):
                local[row, dimension] = 0.0
        for image in range(orbit_image_offsets[orbit], orbit_image_offsets[orbit + 1]):
            basis_start = image_basis_offsets[image]
            for translation in range(translation_count):
                for component in range(component_count):
                    for axis in range(order):
                        row = (
                            3 * image_atoms[image, translation, axis] + components[component, axis]
                        )
                        monomial = 1.0
                        for other in range(order):
                            if other != axis:
                                monomial *= displacement[
                                    3 * image_atoms[image, translation, other]
                                    + components[component, other]
                                ]
                        # Differentiate every slot of the energy term: repeated
                        # coordinates contribute once per slot, with the force sign.
                        value = -factor * monomial
                        basis_row = basis_start + component * dimension_count
                        for dimension in range(dimension_count):
                            local[row, dimension] += value * image_basis[basis_row + dimension]
        for row in range(rows):
            for dimension in range(dimension_count):
                out[row, parameter_start + dimension] += local[row, dimension]


@dataclass(slots=True)
class DesignWorkspace:
    """Mutable per-order/thread scratch owned by one force-design caller.

    scratch[i] has shape (threads, rows, dimensions[i]). Reuse serially;
    sharing an instance across concurrent calls is unsafe.
    """

    rows: int
    dimensions: tuple[int, ...]
    threads: int
    scratch: tuple[np.ndarray, ...]

    def validate(self, rows, dimensions):
        """Require matching dimensions, writable contiguous float64 buffers and sufficient thread
        slots.
        """
        if (rows, dimensions) != (self.rows, self.dimensions):
            raise ValueError("design workspace belongs to different dimensions")
        if len(self.scratch) != len(dimensions):
            raise ValueError("design workspace has inconsistent scratch buffers")
        for array, dimension in zip(self.scratch, dimensions, strict=True):
            if (
                array.shape != (self.threads, rows, dimension)
                or array.dtype != np.float64
                or not array.flags.c_contiguous
                or not array.flags.writeable
            ):
                raise ValueError("design workspace scratch has invalid shape, dtype or storage")
        if get_num_threads() > self.threads:
            raise ValueError("design workspace has fewer slots than active Numba threads")


def allocate_design_workspace(rows, dimensions):
    """Allocate zeroed scratch buffers after checking each per-thread array size."""
    threads = get_num_threads()
    for dimension in dimensions:
        require_allocation("design scratch", (threads, rows, dimension))
    return DesignWorkspace(
        rows,
        dimensions,
        threads,
        tuple(np.zeros((threads, rows, dimension), dtype=np.float64) for dimension in dimensions),
    )


__all__ = ["ForceDesign"]
