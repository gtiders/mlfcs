"""Streamed construction of portable least-squares sufficient statistics."""

from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
from scipy.linalg.blas import dsyrk

from mlfcs.fitting.gram.models import GramStatistics

logger = logging.getLogger(__name__)


class GramAccumulator:
    """Normal-equation sufficient statistics accumulated without storing A."""

    def __init__(self, gram, rhs, target_norm):
        self.gram = gram
        self.rhs = rhs
        self.target_norm = target_norm

    @classmethod
    def from_operator(cls, operator, target, *, metadata=None):
        del cls
        started = perf_counter()
        n_parameters = operator.n_parameters
        gram = np.zeros((n_parameters, n_parameters), dtype=float, order="F")
        rhs = np.zeros(n_parameters, dtype=float)
        target_shaped = np.asarray(target).reshape(operator.force_shape)
        count = len(operator.displacements)
        logger.info(
            f"Accumulating streamed Gram system: {n_parameters} x {n_parameters} "
            f"({gram.nbytes / 1024**2:.1f} MiB) over {count} snapshots, "
            f"backend=Numba design + OpenBLAS Gram"
        )
        design_seconds = 0.0
        gram_seconds = 0.0

        for index in range(count):
            design_started = perf_counter()
            design = operator.design(index)
            design_seconds += perf_counter() - design_started
            force = target_shaped[index].reshape(-1)
            gram_started = perf_counter()
            dsyrk(
                1.0,
                a=design,
                c=gram,
                beta=1.0,
                trans=1,
                lower=0,
                overwrite_c=1,
            )
            rhs += design.T @ force
            gram_seconds += perf_counter() - gram_started
            if logger.isEnabledFor(logging.INFO) and (
                index == 0 or index + 1 == count or (index + 1) % 20 == 0
            ):
                logger.info(
                    f"- Gram structures: {index + 1}/{count}, "
                    f"elapsed={perf_counter() - started:.2f} s"
                )

        upper = np.triu(np.asarray(gram))
        gram = upper + np.triu(upper, 1).T
        logger.info(f"- Streamed Gram system ready in {perf_counter() - started:.2f} s")
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"- Gram phase timing: design={design_seconds:.2f} s, BLAS={gram_seconds:.2f} s"
            )
        target_norm = float(np.vdot(target, target))
        return GramStatistics(
            np.asarray(gram),
            np.asarray(rhs),
            target_norm,
            len(target),
            dict(metadata or {}),
        )


class GramBuilder:
    """Explicit entry point for one-shot, portable Gram construction."""

    @classmethod
    def from_operator(cls, operator, target, *, metadata=None) -> GramStatistics:
        del cls
        return GramAccumulator.from_operator(operator, target, metadata=metadata)
