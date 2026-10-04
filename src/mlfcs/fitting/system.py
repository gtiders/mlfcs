"""Physical force-fitting equations in raw or normal representation."""

from __future__ import annotations

import hashlib
import logging
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from ase import Atoms
from scipy.linalg.blas import dsyrk

from mlfcs._arrays import require_allocation
from mlfcs.cluster_space import ClusterSpace
from mlfcs.core.geometry import PeriodicGeometry
from mlfcs.core.log import get_logger
from mlfcs.fitting.design import ForceDesign
from mlfcs.fitting.solve import FitSolver
from mlfcs.force_constants import ForceConstants
from mlfcs.mapping import ClusterMap

logger = get_logger(__name__)


def _readonly(values: object, shape: tuple[int, ...]) -> np.ndarray:
    """Copy physical equation data into finite, shape-checked readonly C-contiguous float64 storage."""
    require_allocation("fit buffer", shape)
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"fit array must be finite and have shape {shape}")
    result.setflags(write=False)
    return result


def _bytes(values: np.ndarray) -> bytes:
    """Encode equation entries in a fixed big-endian byte order for stable hashing."""
    return values.astype(values.dtype.newbyteorder(">"), copy=False).tobytes()


@dataclass(frozen=True, slots=True, init=False)
class FitSystem:
    """Immutable physical force-fitting equations in one of two representations.

    Parameters
    ----------
    cluster_map : ClusterMap
        Reference supercell and primitive parameter model. ForceDesign requires
        full exact structural rank; the completed system retains only its
        ClusterSpace and equations, not the map or training structures.
    structures : iterable of ase.Atoms
        Nonempty displaced snapshots with matching atom order, PBC and reference
        cell within symprec. Forces must already be stored in ASE calculators.
    representation : {'normal', 'raw'}, default 'normal'
        normal streams H = A.T @ A, g = A.T @ f and f.T @ f; raw retains A and f.
        This fixes the built-in solver to MINRES or LSMR respectively.

    Notes
    -----
    Displacements are minimum-image Cartesian vectors in angstrom. Force rows
    are atom-major x/y/z in eV/angstrom; columns follow canonical primitive
    parameters. ForceDesign includes force signs, Taylor factorials and images.
    Construction does not evaluate calculators, subtract mean forces, weight
    samples or normalize columns. Public arrays are readonly and unscaled.
    Solving normalizes temporary data and returns physical coefficients.

    Raises
    ------
    ValueError
        Samples, representation or stored force values are inconsistent.
    TypeError
        The map or sample container has an unsupported type.
    AliasingError
        The supercell cannot distinguish all primitive parameters.
    OverflowError
        An actual equation/workspace allocation exceeds representable capacity.

    Examples
    --------
    >>> system = FitSystem(mapping, structures, representation='raw')
    >>> model = system.solve(atol=1e-8, btol=1e-8)
    >>> system.rmse(model)  # doctest: +SKIP
    """

    cluster_space: ClusterSpace
    representation: str
    _matrix: np.ndarray
    _rhs: np.ndarray
    force_squared_norm: float
    n_equations: int
    n_structures: int

    def __init__(
        self, cluster_map: ClusterMap, structures: Iterable[Atoms], *, representation="normal"
    ):
        """Ingest stored-force snapshots once and initialize unscaled physical equations."""
        if not isinstance(cluster_map, ClusterMap):
            raise TypeError("cluster_map must be a ClusterMap")
        if representation not in ("raw", "normal"):
            raise ValueError("representation must be 'raw' or 'normal'")
        started = perf_counter()
        arrays = _build_equations(cluster_map, structures, representation)
        self._initialize(cluster_map.cluster_space, representation, *arrays)
        logger.info(
            "Fit-system complete: representation=%s structures=%d equations=%d "
            "parameters=%d shape=%s equation_bytes=%d elapsed_s=%.2f",
            self.representation,
            self.n_structures,
            self.n_equations,
            self.n_parameters,
            self._matrix.shape,
            self._matrix.nbytes + self._rhs.nbytes,
            perf_counter() - started,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Fit-system identity: fit_system_fingerprint=%s", self.fingerprint)

    def _initialize(
        self,
        cluster_space,
        representation,
        matrix,
        rhs,
        force_squared_norm,
        n_equations,
        n_structures,
    ):
        """Validate internal equation statistics and capture independent readonly buffers."""
        if not isinstance(cluster_space, ClusterSpace):
            raise TypeError("cluster_space must be a ClusterSpace")
        if representation not in ("raw", "normal"):
            raise ValueError("representation must be 'raw' or 'normal'")
        n_equations, n_structures = int(n_equations), int(n_structures)
        if n_equations < 0 or n_structures < 0:
            raise ValueError("fit-system counts must be nonnegative")
        count = cluster_space.n_parameters
        shape = (n_equations, count) if representation == "raw" else (count, count)
        matrix = _readonly(matrix, shape)
        rhs = _readonly(rhs, (n_equations if representation == "raw" else count,))
        force_squared_norm = float(force_squared_norm)
        if not np.isfinite(force_squared_norm) or force_squared_norm < 0.0:
            raise ValueError("force_squared_norm must be finite and nonnegative")
        if representation == "normal":
            if not np.array_equal(matrix, matrix.T):
                raise ValueError("normal matrix must be exactly symmetric")
            diagonal = np.diag(matrix)
            if np.any(diagonal < 0.0):
                raise ValueError("normal matrix diagonal must be nonnegative")
            if np.any(rhs[diagonal == 0.0] != 0.0):
                raise ValueError("a zero normal-matrix column has a nonzero right-hand side")
        for name, value in (
            ("cluster_space", cluster_space),
            ("representation", representation),
            ("_matrix", matrix),
            ("_rhs", rhs),
            ("force_squared_norm", force_squared_norm),
            ("n_equations", n_equations),
            ("n_structures", n_structures),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def _from_equations(
        cls,
        cluster_space,
        representation,
        matrix,
        rhs,
        force_squared_norm,
        n_equations,
        n_structures,
    ):
        """Construct validated internal equations without re-ingesting ASE snapshots."""
        result = object.__new__(cls)
        result._initialize(
            cluster_space,
            representation,
            matrix,
            rhs,
            force_squared_norm,
            n_equations,
            n_structures,
        )
        return result

    def __reduce__(self):
        """Serialize equations and primitive model for validated restoration."""
        return _restore_fit_system, (
            self.cluster_space,
            self.representation,
            self._matrix,
            self._rhs,
            self.force_squared_norm,
            self.n_equations,
            self.n_structures,
        )

    def _require_representation(self, representation):
        """Raise ValueError if data is requested from the wrong equation representation."""
        if self.representation != representation:
            raise ValueError(f"this data is only available for representation={representation!r}")

    @property
    def design_matrix(self):
        """Readonly unscaled A, shape (n_equations, n_parameters); raw only, otherwise ValueError."""
        self._require_representation("raw")
        return self._matrix

    @property
    def forces(self):
        """Readonly unscaled force vector f, shape (n_equations,); raw only, otherwise ValueError."""
        self._require_representation("raw")
        return self._rhs

    @property
    def normal_matrix(self):
        """Readonly H = A.T @ A, shape (n_parameters, n_parameters); normal only."""
        self._require_representation("normal")
        return self._matrix

    @property
    def normal_rhs(self):
        """Readonly g = A.T @ f, shape (n_parameters,); normal only."""
        self._require_representation("normal")
        return self._rhs

    @property
    def n_parameters(self):
        """Canonical parameter count across every fitted cluster-space order."""
        return self.cluster_space.n_parameters

    @property
    def unobserved_parameters(self):
        """Indices of columns identically absent from these training equations.

        Uses exact zeros, not a numerical-rank tolerance. Empty output does not
        prove that the training matrix has full numerical rank.
        """
        if self.representation == "normal":
            missing = np.diag(self._matrix) == 0.0
        else:
            missing = ~np.any(self._matrix != 0.0, axis=0)
        return tuple(int(index) for index in np.flatnonzero(missing))

    @property
    def fingerprint(self):
        """Hash model identity, representation, counts and physical equation contents."""
        digest = hashlib.sha256()
        digest.update(self.cluster_space.fingerprint.encode("ascii"))
        digest.update(self.representation.encode("ascii"))
        digest.update(
            struct.pack(">qqd", self.n_equations, self.n_structures, self.force_squared_norm)
        )
        digest.update(_bytes(self._matrix))
        digest.update(_bytes(self._rhs))
        return digest.hexdigest()

    def parameter_name(self, parameter: int) -> str:
        """Return the physical layout address of one packed parameter."""
        parameter = int(parameter)
        if not 0 <= parameter < self.n_parameters:
            raise IndexError("parameter is outside the cluster space")
        offsets = self.cluster_space.parameter_offsets
        orbit_index = int(np.searchsorted(offsets, parameter, side="right") - 1)
        component = parameter - int(offsets[orbit_index])
        for block in self.cluster_space.blocks:
            if block.parameters.start <= parameter < block.parameters.stop:
                return (
                    f"FC{block.order} orbit {orbit_index - block.orbits.start} "
                    f"component {component}"
                )
        raise RuntimeError("cluster-space parameter layout is inconsistent")

    def solve(self, **options) -> ForceConstants:
        """Return ForceConstants using temporary column scaling and the fixed solver.

        Parameters
        ----------
        **options
            normal accepts rtol=1e-8 and maxiter=1000 for MINRES. raw accepts
            atol=1e-8, btol=1e-8, conlim=1e8 and maxiter=1000 for LSMR.

        Returns
        -------
        model : ForceConstants
            All fitted orders with physical parameters in canonical layout.

        Raises
        ------
        UnobservedParameterError
            One or more parameter columns are identically zero.
        RuntimeError
            The selected algorithm fails its stopping conditions.
        TypeError
            An option is unsupported by the selected algorithm.

        Notes
        -----
        Only solver temporaries are normalized; exposed data is unchanged.
        There is no algorithm selector, regularization or automatic solver fallback.
        """
        started = perf_counter()
        logger.info(
            "Fit solve started: representation=%s structures=%d equations=%d "
            "parameters=%d orders=%s",
            self.representation,
            self.n_structures,
            self.n_equations,
            self.n_parameters,
            self.cluster_space.orders,
        )
        model = self.force_constants(FitSolver(self).solve(**options))
        if logger.isEnabledFor(logging.INFO):
            try:
                residual = self.residual(model.parameters())
                if not self.n_equations:
                    raise ValueError("no training equations for RMSE")
                rmse = residual / np.sqrt(self.n_equations)
                relative = (
                    residual / np.sqrt(self.force_squared_norm)
                    if self.force_squared_norm
                    else (0.0 if residual == 0.0 else float("inf"))
                )
                logger.info(
                    "Fit complete: training_force_rmse=%.12g eV/angstrom "
                    "training_relative_force_error=%.8g elapsed_s=%.2f",
                    rmse,
                    relative,
                    perf_counter() - started,
                )
            except ValueError as error:
                # Diagnostics must not turn a successful solve into a failure.
                logger.warning(
                    "Fit complete: quality_summary_unavailable=%s elapsed_s=%.2f",
                    error,
                    perf_counter() - started,
                )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Fit identity: fit_system_fingerprint=%s force_constants_fingerprint=%s",
                self.fingerprint,
                model.fingerprint,
            )
        return model

    def _parameters(self, values):
        """Validate a physical parameter vector or a complete model with matching space identity."""
        if isinstance(values, ForceConstants):
            if values.cluster_space.fingerprint != self.cluster_space.fingerprint:
                raise ValueError("force constants use a different cluster space")
            if values.orders != self.cluster_space.orders:
                raise ValueError("force constants must contain every fitted order")
            values = values.parameters()
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n_parameters,):
            raise ValueError(
                f"parameters must have shape {(self.n_parameters,)}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("parameters must be finite")
        return values

    def force_constants(self, parameters) -> ForceConstants:
        """Bind an external solver's physical parameter vector to all fitted orders.

        parameters has shape (n_parameters,) in canonical order and must be finite.
        A matching complete ForceConstants is also accepted. Any external scaling
        must be undone by the caller first; no scaling or solving happens here.
        """
        values = self._parameters(parameters)
        return ForceConstants(
            self.cluster_space,
            {block.order: values[block.parameters] for block in self.cluster_space.blocks},
        )

    def residual(self, model_or_parameters):
        """Return ||A @ theta - f|| in physical force units for a model or vector.

        Normal data uses theta.T @ H @ theta - 2*theta.T @ g + f.T @ f; roundoff
        cancellation near zero is clipped. Negative squared residual beyond that
        roundoff bound raises ValueError, as do incompatible or nonfinite values.
        """
        values = self._parameters(model_or_parameters)
        if self.representation == "raw":
            residual = float(np.linalg.norm(self._matrix @ values - self._rhs))
            if not np.isfinite(residual):
                raise ValueError("fit-system residual is not finite")
            return residual
        model = float(values @ self._matrix @ values)
        cross = float(values @ self._rhs)
        squared = model - 2.0 * cross + self.force_squared_norm
        cancellation = (
            32.0
            * np.finfo(float).eps
            * (abs(model) + 2.0 * abs(cross) + abs(self.force_squared_norm))
        )
        if not np.isfinite(squared) or not np.isfinite(cancellation):
            raise ValueError("fit-system residual is not finite")
        # Normal statistics can lose a tiny residual by subtracting large
        # terms; clip only within the explicit floating-roundoff allowance.
        if abs(squared) <= cancellation:
            squared = 0.0
        if squared < 0.0:
            raise ValueError(
                "fit-system residual squared is negative beyond roundoff; normal statistics are inconsistent"
            )
        return float(np.sqrt(squared))

    def rmse(self, model_or_parameters):
        """Return physical force residual divided by sqrt(n_equations), or zero for empty zero data."""
        residual = self.residual(model_or_parameters)
        if self.n_equations:
            return residual / np.sqrt(self.n_equations)
        if residual != 0.0:
            raise ValueError("fit system has a nonzero residual but no equations")
        return 0.0

    def relative_error(self, model_or_parameters):
        """Return ||A @ theta - f|| / ||f|| for physical parameters.

        For zero reference force norm, return zero for exact zero residual and
        infinity otherwise. Input compatibility follows residual.
        """
        residual = self.residual(model_or_parameters)
        if self.force_squared_norm > 0.0:
            return residual / np.sqrt(self.force_squared_norm)
        return 0.0 if residual == 0.0 else float("inf")

    def to_normal(self):
        """Return normal statistics of raw physical equations; a normal system returns itself.

        Conversion allocates H and g but preserves counts and force_squared_norm.
        It loses row-level data; the raw source remains unchanged.
        """
        if self.representation == "normal":
            return self
        matrix = _normal_matrix(self._matrix)
        return type(self)._from_equations(
            self.cluster_space,
            "normal",
            matrix,
            self._matrix.T @ self._rhs,
            self.force_squared_norm,
            self.n_equations,
            self.n_structures,
        )

    def __add__(self, other):
        """Merge same-space, same-representation systems without modifying either input.

        Normal statistics add; raw equations concatenate rows left then right.
        Mixed representations raise ValueError and require explicit to_normal().
        Other operand types return NotImplemented.
        """
        if not isinstance(other, FitSystem):
            return NotImplemented
        if self.cluster_space.fingerprint != other.cluster_space.fingerprint:
            raise ValueError("fit systems use different cluster spaces")
        if self.representation != other.representation:
            raise ValueError(
                "fit systems use different representations; convert raw with to_normal() explicitly"
            )
        if self.representation == "normal":
            matrix, rhs = self._matrix + other._matrix, self._rhs + other._rhs
        else:
            require_allocation(
                "merged raw design", (self.n_equations + other.n_equations, self.n_parameters)
            )
            matrix = np.concatenate((self._matrix, other._matrix))
            rhs = np.concatenate((self._rhs, other._rhs))
        return type(self)._from_equations(
            self.cluster_space,
            self.representation,
            matrix,
            rhs,
            self.force_squared_norm + other.force_squared_norm,
            self.n_equations + other.n_equations,
            self.n_structures + other.n_structures,
        )


def _restore_fit_system(*state):
    """Restore serialized physical equations through the validated internal constructor."""
    return FitSystem._from_equations(*state)


def _sample(
    cluster_map: ClusterMap,
    atoms: Atoms,
    index: int,
    supercell_positions: np.ndarray,
    geometry: PeriodicGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one snapshot and return minimum-image displacements and stored forces.

    Both outputs have shape (n_atoms, 3), in angstrom and eV/angstrom respectively.
    Cell and atom order must match the map; calculator access uses
    allow_calculation=False, so missing forces fail without launching a calculation.
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(f"training structure {index} is not an ASE Atoms object")
    supercell = cluster_map
    if not np.array_equal(atoms.numbers, supercell.atomic_numbers):
        raise ValueError(f"training structure {index} has a different atom sequence")
    if not np.array_equal(atoms.pbc, np.ones(3, dtype=bool)):
        raise ValueError(f"training structure {index} must be periodic in all directions")
    cell = np.asarray(atoms.cell, dtype=np.float64)
    cell_residual = float(np.max(np.linalg.norm(cell - supercell.cell, axis=1)))
    symprec = cluster_map.cluster_space.symprec
    if cell_residual >= symprec:
        raise ValueError(
            f"training structure {index} has cell residual {cell_residual:.10g} Å, "
            f"not below symprec {symprec:.10g} Å"
        )
    displacement, _ = geometry.minimum_image(atoms.positions - supercell_positions)
    if not np.all(np.isfinite(displacement)):
        raise ValueError(f"training structure {index} contains invalid positions")
    if atoms.calc is None:
        raise ValueError(f"training structure {index} has no stored ASE forces")
    forces = atoms.calc.get_property("forces", atoms, allow_calculation=False)
    if forces is None:
        raise ValueError(f"training structure {index} has no stored ASE forces")
    forces = np.asarray(forces, dtype=np.float64)
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError(f"training structure {index} contains invalid forces")
    return displacement, forces


def _normal_matrix(design):
    """Allocate A.T @ A with upper-triangle DSYRK and return an exactly symmetric float64 matrix."""
    require_allocation("fit normal matrix", (design.shape[1], design.shape[1]))
    matrix = dsyrk(1.0, a=design, trans=1, lower=0)
    return _symmetrize(matrix)


def _symmetrize(matrix):
    """Copy the stored upper triangle to both halves without averaging or using the lower triangle."""
    upper = np.triu(np.asarray(matrix))
    return upper + np.triu(upper, 1).T


def _build_equations(cluster_map, structures, representation):
    """Consume snapshots once, compiling one design and reusing its mutable workspace.

    Normal mode streams upper-triangle Gram sums and RHS; raw mode stacks
    physical design/force rows. Return matrix, rhs, force_squared_norm,
    n_equations and n_structures. No snapshots or normalized equations are kept.
    """
    if isinstance(structures, (Atoms, np.ndarray)):
        raise TypeError("FitSystem requires an iterable of ASE Atoms")
    started = perf_counter()
    logger.info(
        "Fit-system construction started: representation=%s supercell_atoms=%d "
        "orders=%s parameters=%d equations_per_structure=%d",
        representation,
        cluster_map.n_atoms,
        cluster_map.cluster_space.orders,
        cluster_map.cluster_space.n_parameters,
        3 * cluster_map.n_atoms,
    )
    design = ForceDesign(cluster_map)
    workspace = design.allocate_workspace()
    parameters = design.n_parameters
    if representation == "normal":
        require_allocation("fit normal matrix", (parameters, parameters))
        require_allocation("fit right hand side", (parameters,))
        matrix = np.zeros((parameters, parameters), dtype=np.float64, order="F")
        rhs = np.zeros(parameters, dtype=np.float64)
    else:
        matrices, force_vectors = [], []
    force_squared_norm = 0.0
    count = 0
    supercell_positions = cluster_map.scaled_positions @ cluster_map.cell
    geometry = PeriodicGeometry(cluster_map.cell)
    for count, atoms in enumerate(structures, start=1):
        displacement, forces = _sample(cluster_map, atoms, count - 1, supercell_positions, geometry)
        values = design.matrix(displacement, workspace=workspace)
        flattened = forces.reshape(-1)
        if representation == "normal":
            # DSYRK updates only the upper triangle; mirror it after streaming
            # all frames, rather than treating the unused lower triangle as data.
            dsyrk(1.0, a=values, c=matrix, beta=1.0, trans=1, lower=0, overwrite_c=1)
            rhs += values.T @ flattened
        else:
            require_allocation("raw design", (count * design.rows, parameters))
            require_allocation("raw forces", (count * design.rows,))
            matrices.append(values)
            force_vectors.append(flattened.copy())
        force_squared_norm += float(flattened @ flattened)
        if count == 1 or count % 10 == 0:
            elapsed = perf_counter() - started
            logger.info(
                "Fit-system accumulation: structures=%d equations=%d elapsed_s=%.2f "
                "structures_per_s=%.3g",
                count,
                count * design.rows,
                elapsed,
                count / elapsed if elapsed else float("inf"),
            )
    if count == 0:
        raise ValueError("at least one training structure is required")
    if representation == "normal":
        matrix = _symmetrize(matrix)
    else:
        matrix, rhs = np.concatenate(matrices), np.concatenate(force_vectors)
    logger.info(
        "Fit equations accumulated: representation=%s parameters=%d structures=%d elapsed_s=%.2f",
        representation,
        parameters,
        count,
        perf_counter() - started,
    )
    return matrix, rhs, force_squared_norm, count * design.rows, count


__all__ = ["FitSystem"]
