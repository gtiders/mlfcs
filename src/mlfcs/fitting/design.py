"""Taylor force designs in physical and free acoustic parameter coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from time import perf_counter

import numpy as np
from llvmlite import ir
from numba import get_num_threads, get_thread_id, njit, prange, types
from numba.extending import intrinsic

from mlfcs.foundation.arrays import as_int64_array, readonly, require_allocation
from mlfcs.foundation.tensors import rotate_basis
from mlfcs.mapping import ClusterMap

# Translations per accumulation block: small enough that one thread's
# (block, 3, dimension) buffer stays in L1.
_TRANSLATION_BLOCK = 16

# Vector width of the accumulation kernel. Numba's loop vectorizer cannot pack
# ``block[target + d] += value * basis[base + d]`` because the target index is a
# runtime value, so the lane group is emitted as explicit LLVM IR instead. Eight
# lanes map to one AVX-512 register; hosts without AVX-512 legalize the same IR
# into two 256-bit halves.
_LANES = 8


@intrinsic
def _axpy8(typingctx, target, offset, source, source_offset, value):
    """Emit one eight-lane fused multiply-add into ``target``.

    Writes ``target[offset + i] += value * source[source_offset + i]`` for
    ``i`` in ``0, ..., 7`` as one ``<8 x double>`` load, broadcast, fused
    multiply-add and store. The data pointers come from Numba's array model and
    the GEP carries the element type explicitly, so no integer/pointer round
    trip occurs and both typed and opaque pointer IR are supported. Alignment
    is declared as one double because the offsets are arbitrary; an aligned
    declaration would fault on the JIT.
    """
    signature = types.void(target, offset, source, source_offset, value)

    def codegen(context, builder, sig, args):
        """Emit one unaligned eight-double load, fused multiply-add and store."""
        target_array, target_offset, source_array, source_offset, scalar = args
        double = ir.DoubleType()
        vector = ir.VectorType(double, _LANES)
        target_ptr = context.make_array(sig.args[0])(
            context, builder, value=target_array
        ).data
        source_ptr = context.make_array(sig.args[2])(
            context, builder, value=source_array
        ).data
        target_ptr = builder.gep(target_ptr, [target_offset], source_etype=double)
        source_ptr = builder.gep(source_ptr, [source_offset], source_etype=double)
        if not target_ptr.type.is_opaque:
            target_ptr = builder.bitcast(target_ptr, ir.PointerType(vector))
            source_ptr = builder.bitcast(source_ptr, ir.PointerType(vector))
        accumulated = builder.load(target_ptr, typ=vector, align=8)
        factor = ir.Constant(vector, ir.Undefined)
        for lane in range(_LANES):
            factor = builder.insert_element(
                factor, scalar, ir.Constant(ir.IntType(32), lane)
            )
        name = f"llvm.fma.v{_LANES}f64"
        fused = builder.module.globals.get(name)
        if fused is None:
            fused = ir.Function(
                builder.module,
                ir.FunctionType(vector, [vector, vector, vector]),
                name=name,
            )
        product = builder.call(
            fused,
            [factor, builder.load(source_ptr, typ=vector, align=8), accumulated],
        )
        builder.store(product, target_ptr, align=8)

    return signature, codegen


@dataclass(frozen=True, slots=True)
class OrderDesign:
    """Cartesian tensor actions defining one order's Taylor force contribution.

    Rotated orbit bases and supercell translations determine how each physical
    parameter contributes to atomic forces. The Taylor energy factor is
    ``1/p!``; differentiating every tensor slot includes repeated coordinates.

    Compiled representation
    -----------------------
    components is (3**p, p) in C-order direction enumeration. Orbit offsets
    index image ranges; parameter starts refer to the global ClusterSpace.
    image_atoms is (images, translations, p); flattened image_basis stores
    (3**p, orbit_dimension) blocks delimited by image_basis_offsets.
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
            object.__setattr__(self, name, as_int64_array(array, name=name))
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
    image_count = sum(len(space.orbits[i].labels) for i in orbit_indices)
    basis_count = sum(
        len(space.orbits[i].labels) * 3**order * space.orbits[i].dimension for i in orbit_indices
    )
    require_allocation("image atoms", (image_count, len(translations), order))
    require_allocation("image basis", (basis_count,))
    require_allocation("image basis offsets", (image_count + 1,))
    require_allocation("orbit offsets", (len(orbit_indices) + 1,))
    require_allocation("tensor components", (3**order, order))
    translations = as_int64_array(translations, name="mapping translations")
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
        for image in range(len(orbit.labels)):
            atoms = cluster_map.map_labels(orbit.labels[image], translations).T.copy()
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
    """Linear Taylor force model for a mapped periodic supercell.

    ``matrix`` maps canonical physical parameters to Cartesian forces for one
    displacement frame. ``iter_fit_blocks`` expresses the same observations in
    free acoustic coordinates when ASR is enabled.

    Parameters
    ----------
    cluster_map : ClusterMap
        Supercell relation retained by reference. The map already guarantees
        that the supercell distinguishes all primitive parameters.

    Notes
    -----
    Rows are atom-major x/y/z forces; physical columns follow ClusterSpace parameters.
    Workspace is caller-owned and mutable, separate from readonly design data.
    """

    __slots__ = ("cluster_map", "orders", "rows")

    def __init__(self, cluster_map: ClusterMap):
        """Construct the force model for a structurally identifiable supercell."""
        started = perf_counter()

        self.cluster_map = cluster_map
        self.rows = 3 * len(cluster_map.atomic_numbers)
        require_allocation("force design", (self.rows, self.n_parameters))
        translations = cluster_map.translation_representatives
        self.orders = tuple(
            _compile_order(cluster_map, order, translations)
            for order in cluster_map.cluster_space.orders
        )
        for order, compiled in zip(cluster_map.cluster_space.orders, self.orders, strict=True):
            print(
                f"Force design FC{order}: images {len(compiled.image_atoms)}, "
                f"basis {compiled.image_basis.nbytes / 2**20:.2f} MiB"
            )
        print(
            f"Force design ready: rows {self.rows}, columns {self.n_parameters}, "
            f"supercell atoms {cluster_map.n_atoms}, orders {cluster_map.cluster_space.orders}, "
            f"{perf_counter() - started:.2f} s"
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
        workspace = allocate_design_workspace(tuple(order.max_dimension for order in self.orders))

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
        workspace.validate(tuple(order.max_dimension for order in self.orders))
        result = np.zeros((self.rows, self.n_parameters), dtype=np.float64)
        for order, accumulator in zip(self.orders, workspace.accumulator, strict=True):
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
                accumulator,
                workspace.block_length,
            )
        return result

    def iter_fit_blocks(self, displacement, *, rows_per_block=64):
        """Yield bounded force-row blocks in the cluster space's fitting coordinates.

        Acoustic-enabled spaces transform canonical rows through their sparse
        triangular lift. The temporary canonical design contains at most
        ``rows_per_block`` force rows, rather than one entire snapshot.
        """
        values = np.ascontiguousarray(displacement, dtype=float).reshape(-1)
        if values.shape != (self.rows,) or not np.all(np.isfinite(values)):
            raise ValueError("invalid design displacements")
        if rows_per_block < 1:
            raise ValueError("rows_per_block must be positive")
        space = self.cluster_map.cluster_space
        workspace = allocate_design_workspace(tuple(order.max_dimension for order in self.orders))
        # Accumulate every force row once: the kernel walks all images and
        # translations regardless of the requested row window, so splitting the
        # accumulation per block would repeat that work for every block.
        canonical = np.zeros((self.rows, self.n_parameters))
        for order, work in zip(self.orders, workspace.accumulator, strict=True):
            accumulate_design(
                values,
                canonical,
                order.orbit_image_offsets,
                order.orbit_parameter_starts,
                order.orbit_dimensions,
                order.image_atoms,
                order.image_basis,
                order.image_basis_offsets,
                order.components,
                order.factor,
                work,
                workspace.block_length,
            )
        for start in range(0, self.rows, rows_per_block):
            count = min(rows_per_block, self.rows - start)
            reduced = np.empty((count, space.n_free_parameters))
            offset = 0
            for block in space.blocks:
                coordinates = space.acoustic_coordinates(block.order)
                observed = canonical[start : start + count, block.parameters]
                transformed = (
                    observed if coordinates is None else coordinates.restrict_rows(observed)
                )
                reduced[:, offset : offset + transformed.shape[1]] = transformed
                offset += transformed.shape[1]
            yield slice(start, start + count), reduced


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
    accumulator,
    translation_block,
    row_start=0,
):
    """Accumulate one order into a physical force-design matrix in place.

    Displacement is atom-major Cartesian float64; out has shape (rows, total
    parameters). Compiled ragged offsets delimit images and flattened bases.
    Each tensor index contributes -1/p! times the product of all other index
    displacements.

    Contributions are grouped per image and tensor axis so that one translated
    basis block is read once per translation block and each output row is
    written once per image and axis, instead of once per component. Accumulator
    holds one flat ``(translation block, 3, dimension)`` region per thread: the
    kernel accumulates into it while sweeping components, then adds the
    finished rows to ``out``. ``translation_block`` is the block length that
    region was allocated for. Orbits write disjoint parameter columns, so prange
    needs no atomics. Validated metadata admits all indices.
    """
    orbit_count = orbit_dimensions.size
    translation_count = image_atoms.shape[1]
    order = image_atoms.shape[2]
    component_count = components.shape[0]
    rows = out.shape[0]
    block_length = translation_block
    for orbit in prange(orbit_count):
        block = accumulator[get_thread_id()]
        dimension_count = orbit_dimensions[orbit]
        parameter_start = orbit_parameter_starts[orbit]
        for image in range(orbit_image_offsets[orbit], orbit_image_offsets[orbit + 1]):
            basis_start = image_basis_offsets[image]
            # One axis at a time: every component contributes to exactly one of
            # the three force rows of that slot, so the row can be summed before
            # it is written.
            for axis in range(order):
                for start in range(0, translation_count, block_length):
                    stop = min(start + block_length, translation_count)
                    width = stop - start
                    for entry in range(width * 3 * dimension_count):
                        block[entry] = 0.0
                    for component in range(component_count):
                        basis_row = basis_start + component * dimension_count
                        direction = components[component, axis]
                        for local in range(width):
                            translation = start + local
                            monomial = 1.0
                            for other in range(order):
                                if other != axis:
                                    monomial *= displacement[
                                        3 * image_atoms[image, translation, other]
                                        + components[component, other]
                                    ]
                            value = -factor * monomial
                            target = (local * 3 + direction) * dimension_count
                            for dimension in range(0, dimension_count - (_LANES - 1), _LANES):
                                _axpy8(
                                    block,
                                    target + dimension,
                                    image_basis,
                                    basis_row + dimension,
                                    value,
                                )
                            for dimension in range(
                                dimension_count - dimension_count % _LANES, dimension_count
                            ):
                                block[target + dimension] += (
                                    value * image_basis[basis_row + dimension]
                                )
                    for local in range(width):
                        translation = start + local
                        atom = image_atoms[image, translation, axis]
                        for direction in range(3):
                            row = 3 * atom + direction - row_start
                            if row < 0 or row >= rows:
                                continue
                            target = (local * 3 + direction) * dimension_count
                            for dimension in range(dimension_count):
                                out[row, parameter_start + dimension] += block[target + dimension]


@dataclass(slots=True)
class DesignWorkspace:
    """Mutable per-order/thread accumulation buffer owned by one design caller.

    Orbits of one order receive a flat ``(threads, translation block * 3 *
    dimension)`` buffer; the kernel streams one translated basis block through
    it and writes finished force rows. A flat layout keeps the accumulation
    loop address arithmetic simple enough to vectorize. Reuse serially; sharing
    an instance across concurrent calls is unsafe.
    """

    dimensions: tuple[int, ...]
    threads: int
    block_length: int
    accumulator: tuple[np.ndarray, ...]

    def validate(self, dimensions):
        """Require matching dimensions, writable contiguous float64 buffers and enough slots."""
        if dimensions != self.dimensions:
            raise ValueError("design workspace belongs to different dimensions")
        if len(self.accumulator) != len(dimensions):
            raise ValueError("design workspace has inconsistent accumulator buffers")
        for array, dimension in zip(self.accumulator, dimensions, strict=True):
            if (
                array.shape != (self.threads, self.block_length * 3 * dimension)
                or array.dtype != np.float64
                or not array.flags.c_contiguous
                or not array.flags.writeable
            ):
                raise ValueError("design workspace has invalid shape, dtype or storage")
        if get_num_threads() > self.threads:
            raise ValueError("design workspace has fewer slots than active Numba threads")


def allocate_design_workspace(dimensions):
    """Allocate zeroed accumulation buffers after checking each per-thread array size."""
    threads = get_num_threads()
    block_length = _TRANSLATION_BLOCK
    for dimension in dimensions:
        require_allocation("design accumulator", (threads, block_length * 3 * dimension))
    return DesignWorkspace(
        dimensions,
        threads,
        block_length,
        tuple(
            np.zeros((threads, block_length * 3 * dimension), dtype=np.float64)
            for dimension in dimensions
        ),
    )


__all__ = ["ForceDesign"]
