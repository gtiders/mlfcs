"""Column-scaled direct and iterative solvers for force-fitting systems."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import lstsq
from scipy.sparse.linalg import minres

from mlfcs.foundation.errors import UnobservedParameterError
from mlfcs.foundation.log import get_logger

if TYPE_CHECKING:
    from mlfcs.fitting.system import FitSystem

logger = get_logger(__name__)


def _maxiter(value):
    """Require a positive integer iteration limit through the integer index protocol."""
    value = operator.index(value)
    if value < 1:
        raise ValueError("maxiter must be a positive integer")
    return value


class FitSolver:
    """Internal solver for a FitSystem; representation fixes the algorithm.

    Normal systems use diagonally scaled MINRES; raw systems use column-scaled
    dense least squares. Scaling is temporary, public physical equations are
    unchanged, and the returned vector is restored to physical coordinates.
    """

    __slots__ = ("system",)

    def __init__(self, system: FitSystem):
        """Retain the system by reference without constructing normalized equations."""
        self.system = system

    def solve(self, **options) -> np.ndarray:
        """Return physical parameters using the representation's sole built-in algorithm.

        Reject completely unobserved columns with UnobservedParameterError. Options
        are forwarded only to the selected algorithm; incompatible names raise
        TypeError. Nonconvergence raises RuntimeError and nonfinite physical output
        raises ValueError. The system's readonly arrays remain unchanged.
        """
        if self.system.n_parameters == 0:
            if options:
                # Keep route-specific option validation even for a zero-dimensional model.
                return (
                    self._normal(**options)
                    if self.system.representation == "normal"
                    else self._raw(**options)
                )
            return np.empty(0, dtype=float)
        missing = self.system.unobserved_parameters
        if missing:
            shown = ", ".join(
                (
                    f"acoustic coordinate {index}"
                    if self.system.cluster_space.asr
                    else self.system.cluster_space.parameter_name(index)
                )
                for index in missing[:12]
            )
            remainder = "" if len(missing) <= 12 else f", and {len(missing) - 12} more"
            raise UnobservedParameterError(
                f"{len(missing)} parameters are absent from the training design: "
                f"{shown}{remainder}. Add structurally different displacements or merge another "
                "FitSystem. Regularization cannot recover a parameter absent from the design."
            )
        if self.system.representation == "normal":
            parameters = self._normal(**options)
        else:
            parameters = self._raw(**options)
        if not np.all(np.isfinite(parameters)):
            raise ValueError("physical fitted parameters are not finite")
        return parameters

    def _normal(self, *, rtol=1e-8, maxiter=1000):
        """Solve S H S z = S g by MINRES and return S z in physical coordinates.

        S = diag(1/sqrt(diag(H))). rtol is positive, maxiter a positive integer.
        Scaled equations are temporary; stopping uses SciPy MINRES's relative rule,
        not the physical force residual. Nonzero stop info raises RuntimeError.
        """
        if not np.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("rtol must be positive and finite")
        maxiter = _maxiter(maxiter)
        logger.info(
            "Solver started: algorithm=MINRES normalization=inverse_column_norm "
            "rtol=%.3g maxiter=%d",
            rtol,
            maxiter,
        )
        matrix, rhs = self.system.normal_matrix, self.system.normal_rhs
        if len(rhs) == 0:
            return np.empty(0, dtype=float)
        scale = 1.0 / np.sqrt(np.diag(matrix))
        normal = matrix * scale[:, None] * scale[None, :]
        scaled_rhs = scale * rhs
        if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(scaled_rhs)):
            raise ValueError("normalized normal equations are not finite")
        iterations = 0

        def callback(_value):
            """Count completed MINRES iterations without retaining intermediate solutions."""
            nonlocal iterations
            iterations += 1

        scaled, info = minres(
            normal,
            scaled_rhs,
            x0=np.zeros(len(rhs)),
            rtol=rtol,
            maxiter=maxiter,
            callback=callback,
            check=True,
        )
        parameters = scale * scaled
        residual = float(np.linalg.norm(matrix @ parameters - rhs))
        logger.info(
            "Solver finished: algorithm=MINRES iterations=%d stop_code=%d physical_normal_residual=%.10e",
            iterations,
            info,
            residual,
        )
        if info != 0:
            raise RuntimeError(
                f"MINRES did not converge in {iterations} steps: stop code {info}, "
                f"physical normal residual {residual:.10e}"
            )
        return parameters

    def _raw(self):
        """Solve the column-scaled raw equations by dense least squares.

        The matrix is scaled to unit column norm before LAPACK solves it. The
        rank cutoff is SciPy's machine-precision default; no iterative stopping
        tolerance or condition limit is exposed. Rank-deficient systems return
        the minimum-norm solution in scaled parameter coordinates.
        """
        logger.info("Solver started: algorithm=least_squares normalization=unit_column_norm")
        matrix, forces = self.system.design_matrix, self.system.forces
        # Scale by each maximum before computing norms to avoid squaring large
        # physical coefficients while normalizing columns from different orders.
        maxima = np.max(np.abs(matrix), axis=0)
        normalized = np.array(matrix, dtype=np.float64, order="F", copy=True)
        normalized /= maxima[None, :]
        norms = np.sqrt(np.einsum("ij,ij->j", normalized, normalized))
        normalized /= norms[None, :]
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            scale = (1.0 / norms) / maxima
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError(
                "raw column normalization cannot represent the physical parameter scale"
            )
        scaled, _, rank, singular_values = lstsq(
            normalized,
            forces,
            cond=None,
            overwrite_a=True,
            check_finite=False,
            lapack_driver="gelsd",
        )
        condition = (
            float(singular_values[0] / singular_values[rank - 1]) if rank > 0 else float("inf")
        )
        logger.info(
            "Solver finished: algorithm=least_squares rank=%d/%d scaled_condition_estimate=%.10e",
            rank,
            matrix.shape[1],
            condition,
        )
        return scale * scaled
