"""Physical force-fitting equations in raw or normal representation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg.blas import dsyrk

from mlfcs.cluster_space import ClusterSpace
from mlfcs.dataset import ForceDataset
from mlfcs.fitting.design import ForceDesign
from mlfcs.fitting.solve import FitSolver
from mlfcs.force_constants import ForceConstants
from mlfcs.foundation.arrays import require_allocation
from mlfcs.foundation.log import get_logger

logger = get_logger(__name__)


def _readonly(values: object, shape: tuple[int, ...]) -> np.ndarray:
    """Copy physical equation data into finite, shape-checked readonly C-contiguous float64 storage."""
    require_allocation("fit buffer", shape)
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"fit array must be finite and have shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, init=False)
class FitSystem:
    """Immutable physical force-fitting equations in one of two representations.

    Parameters
    ----------
    dataset : ForceDataset
        Ordered physical displacements and target forces for one mapped
        supercell. ForceDesign requires full structural rank. The completed
        system retains its ClusterSpace and equations, not the input dataset.
    representation : {'normal', 'raw'}, default 'normal'
        normal streams H = A.T @ A, g = A.T @ f and f.T @ f; raw retains A and f.
        This fixes the built-in solver to MINRES or dense least squares respectively.

    Notes
    -----
    Displacements are minimum-image Cartesian vectors in angstrom. Force rows
    are atom-major x/y/z in eV/angstrom; columns follow canonical primitive
    parameters unless ASR preparation is enabled, in which case they follow the
    order-local free acoustic coordinates. ForceDesign includes force signs, Taylor
    factorials and images.
    Construction does not evaluate calculators, subtract mean forces, weight
    samples or normalize columns. Public arrays are readonly and unscaled.
    Solving normalizes temporary data and returns physical coefficients.
    Callers must use matching parameter layouts when merging systems or
    evaluating a ForceConstants model; model identity is not checked.

    Raises
    ------
    ValueError
        Samples, representation or stored force values are inconsistent.
    TypeError
        The input is not a ForceDataset.
    AliasingError
        The supercell cannot distinguish all primitive parameters.
    OverflowError
        An actual equation/workspace allocation exceeds representable capacity.

    Examples
    --------
    >>> system = FitSystem(dataset, representation='raw')
    >>> model = system.solve()
    >>> system.rmse(model)  # doctest: +SKIP
    """

    cluster_space: ClusterSpace
    representation: str
    _matrix: np.ndarray
    _rhs: np.ndarray
    force_squared_norm: float
    n_equations: int
    n_structures: int

    def __init__(self, dataset: ForceDataset, *, representation="normal"):
        """Initialize unscaled physical equations from a prepared force dataset."""
        if not isinstance(dataset, ForceDataset):
            raise TypeError("dataset must be a ForceDataset")
        cluster_map = dataset.cluster_map
        if representation not in ("raw", "normal"):
            raise ValueError("representation must be 'raw' or 'normal'")
        started = perf_counter()
        arrays = _build_equations(dataset, representation)
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
        count = cluster_space.n_free_parameters
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
        """Number of fitted coordinates, reduced by ASR when enabled."""
        return self.cluster_space.n_free_parameters

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

    def solve(self, **options) -> ForceConstants:
        """Return ForceConstants using temporary column scaling and the fixed solver.

        Parameters
        ----------
        **options
            normal accepts rtol=1e-8 and maxiter=1000 for MINRES. raw uses
            dense least squares with a fixed machine-precision rank cutoff and
            accepts no solver options.

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
                residual = self.residual(model)
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
        return model

    def _parameters(self, values):
        """Extract a finite physical vector of the fitted length and order set.

        The caller must supply coefficients in this system's parameter layout;
        cross-model geometry and basis compatibility are not checked.
        """
        if isinstance(values, ForceConstants):
            if values.orders != self.cluster_space.orders:
                raise ValueError("force constants must contain every fitted order")
            if self.cluster_space.asr:
                values = np.concatenate(
                    [
                        self.cluster_space.acoustic_coordinates(order).extract(
                            values.coefficients[order]
                        )
                        for order in self.cluster_space.orders
                    ]
                )
            else:
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
        """Bind an external solver's fitting-coordinate vector to all fitted orders.

        parameters has shape (n_parameters,) in fitting-coordinate order and must be finite.
        Acoustic-enabled systems lift free coordinates to canonical coefficients.
        A matching complete ForceConstants is also accepted. Any external scaling
        must be undone by the caller first; no scaling or solving happens here.
        """
        values = self._parameters(parameters)
        coefficients, offset = {}, 0
        for block in self.cluster_space.blocks:
            coordinates = self.cluster_space.acoustic_coordinates(block.order)
            dimension = (
                coordinates.dimension
                if coordinates is not None
                else block.parameters.stop - block.parameters.start
            )
            local = values[offset : offset + dimension]
            coefficients[block.order] = local if coordinates is None else coordinates.lift(local)
            offset += dimension
        return ForceConstants(self.cluster_space, coefficients)

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
        """Merge same-representation systems without modifying either input.

        Normal statistics add; raw equations concatenate rows left then right.
        Mixed representations raise ValueError and require explicit to_normal().
        Other operand types return NotImplemented.
        The caller is responsible for matching physical parameter layouts.
        """
        if not isinstance(other, FitSystem):
            return NotImplemented
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


def _normal_matrix(design):
    """Allocate A.T @ A with upper-triangle DSYRK and return an exactly symmetric float64 matrix."""
    require_allocation("fit normal matrix", (design.shape[1], design.shape[1]))
    matrix = dsyrk(1.0, a=design, trans=1, lower=0)
    return _symmetrize(matrix)


def _symmetrize(matrix):
    """Copy the stored upper triangle to both halves without averaging or using the lower triangle."""
    upper = np.triu(np.asarray(matrix))
    return upper + np.triu(upper, 1).T


def _build_equations(dataset, representation):
    """Accumulate dataset equations using one design and a reusable workspace.

    Normal mode streams upper-triangle Gram sums and RHS; raw mode stacks
    physical design/force rows. Return matrix, rhs, force_squared_norm,
    n_equations and n_structures. No snapshots or normalized equations are kept.
    """
    cluster_map = dataset.cluster_map
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
    workspace = None if cluster_map.cluster_space.asr else design.allocate_workspace()
    parameters = cluster_map.cluster_space.n_free_parameters
    if representation == "normal":
        require_allocation("fit normal matrix", (parameters, parameters))
        require_allocation("fit right hand side", (parameters,))
        matrix = np.zeros((parameters, parameters), dtype=np.float64, order="F")
        rhs = np.zeros(parameters, dtype=np.float64)
    else:
        matrices, force_vectors = [], []
    force_squared_norm = 0.0
    count = 0
    for count, (displacement, forces) in enumerate(
        zip(dataset.displacements, dataset.forces, strict=True), start=1
    ):
        flattened = forces.reshape(-1)
        blocks = (
            design.iter_fit_blocks(displacement)
            if cluster_map.cluster_space.asr
            else ((slice(0, design.rows), design.matrix(displacement, workspace=workspace)),)
        )
        for row_slice, values in blocks:
            target = flattened[row_slice]
            if representation == "normal":
                if parameters:
                    dsyrk(1.0, a=values, c=matrix, beta=1.0, trans=1, lower=0, overwrite_c=1)
                rhs += values.T @ target
            else:
                require_allocation("raw design", (count * design.rows, parameters))
                require_allocation("raw forces", (count * design.rows,))
                matrices.append(values)
                force_vectors.append(target.copy())
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
