"""Column-normalized MINRES and LSMR; public equations stay in physical units."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg import lsmr, minres

from mlfcs.errors import UnobservedParameterError
from mlfcs.log import get_logger

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
    LSMR. Scaling is temporary, public physical equations are unchanged, and
    the returned vector is restored to the original parameter coordinates.
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
        missing = self.system.unobserved_parameters
        if missing:
            shown = ", ".join(
                self.system.cluster_space.parameter_name(index) for index in missing[:12]
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

    def _raw(self, *, atol=1e-8, btol=1e-8, conlim=1e8, maxiter=1000):
        """Solve A S z approximately equal to f by LSMR and return physical S z.

        Scale columns to unit Euclidean norm using their maxima first to avoid
        squaring huge or tiny physical entries. A is copied for normalization;
        f and public equations retain their physical units. atol/btol control
        LSMR stopping, conlim the scaled condition estimate (zero disables it),
        and maxiter is positive. Stop codes 0/1/2/4/5 are accepted; others raise
        RuntimeError. Nonrepresentable scale raises ValueError. In rank-deficient
        problems the solution minimizes norm in scaled, not physical, coordinates.
        """
        for name, value in (("atol", atol), ("btol", btol), ("conlim", conlim)):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
        maxiter = _maxiter(maxiter)
        logger.info(
            "Solver started: algorithm=LSMR normalization=unit_column_norm "
            "atol=%.3g btol=%.3g conlim=%.3g maxiter=%d",
            atol,
            btol,
            conlim,
            maxiter,
        )
        matrix, forces = self.system.design_matrix, self.system.forces
        # Normalize via column maxima first, avoiding squared physical entries
        # that could overflow or underflow when FC orders have disparate scales.
        maxima = np.max(np.abs(matrix), axis=0)
        normalized = matrix / maxima
        norms = np.sqrt(np.einsum("ij,ij->j", normalized, normalized))
        normalized /= norms
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            scale = (1.0 / norms) / maxima
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError(
                "raw column normalization cannot represent the physical parameter scale"
            )
        result = lsmr(
            normalized,
            forces,
            atol=atol,
            btol=btol,
            conlim=conlim,
            maxiter=maxiter,
        )
        scaled, stop, iterations, residual, normal_residual, _, condition, _ = result
        logger.info(
            "Solver finished: algorithm=LSMR iterations=%d stop_code=%d force_residual=%.10e "
            "scaled_normal_residual=%.10e scaled_condition_estimate=%.10e",
            iterations,
            stop,
            residual,
            normal_residual,
            condition,
        )
        if stop not in (0, 1, 2, 4, 5):
            raise RuntimeError(
                f"LSMR did not converge: stop code {stop}, {iterations} steps, "
                f"force residual {residual:.10e}, scaled condition estimate {condition:.10e}"
            )
        return scale * scaled
