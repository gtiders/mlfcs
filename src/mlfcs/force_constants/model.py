"""Primitive force constants in cluster-space coordinates."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np

from mlfcs.cluster_space import ClusterSpace
from mlfcs.foundation.log import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mlfcs.force_constants.rotation import RotationResult
    from mlfcs.mapping import ClusterMap


@dataclass(frozen=True, slots=True)
class ForceConstants:
    """Immutable primitive force-constant coefficients in physical component coordinates.

    Parameters
    ----------
    cluster_space : ClusterSpace
        Primitive geometry and canonical parameter schema, retained by reference.
    coefficients : mapping of int to array_like
        Present tensor orders and finite physical parameter vectors. Each order
        must match its block dimension. Values are copied into readonly float64
        storage, with units eV/angstrom**order under the ASE energy convention.

    Notes
    -----
    Only declared orders belong to the model; absence does not mean zero.
    The mapping is readonly. Supercell expansion is derived from a ClusterMap
    and is not stored. Parameters name representative Cartesian components,
    rather than lattice-basis generator coefficients.

    Raises
    ------
    ValueError
        Coefficients are empty, nonfinite or have an incompatible length.
    KeyError
        An order is not defined by the cluster space.
    """

    cluster_space: ClusterSpace
    coefficients: Mapping[int, np.ndarray]

    def __post_init__(self) -> None:
        """Validate per-order vectors and capture readonly coefficient copies in an immutable
        mapping.
        """
        if not isinstance(self.cluster_space, ClusterSpace):
            raise TypeError("space must be a ClusterSpace")
        values: dict[int, np.ndarray] = {}
        for key, source in self.coefficients.items():
            order = int(key)
            if order != key:
                raise TypeError("force-constant orders must be integers")
            block = self.cluster_space.block(order)
            array = np.array(source, dtype=np.float64, copy=True, order="C").reshape(-1)
            expected = block.parameters.stop - block.parameters.start
            if len(array) != expected:
                raise ValueError(
                    f"order-{order} coefficients must have length {expected}, got {len(array)}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"order-{order} coefficients contain NaN or infinite values")
            array.setflags(write=False)
            values[order] = array
        if not values:
            raise ValueError("force constants must contain at least one order")
        object.__setattr__(self, "coefficients", MappingProxyType(values))

    @property
    def orders(self) -> tuple[int, ...]:
        """Explicitly present tensor orders in ascending order."""
        return tuple(sorted(self.coefficients))

    def parameters(self, orders: Iterable[int] | None = None) -> np.ndarray:
        """Return a writable packed coefficient copy in ascending-order parameter layout.

        orders defaults to all present orders; an explicit selection must be unique
        and ascending. Missing orders raise KeyError. No normalization is applied.
        """
        selected = self.orders if orders is None else tuple(int(order) for order in orders)
        if tuple(sorted(set(selected))) != selected:
            raise ValueError("orders must be unique and ascending")
        missing = tuple(order for order in selected if order not in self.coefficients)
        if missing:
            raise KeyError(f"force constants do not contain orders {missing}")
        return np.concatenate([self.coefficients[order] for order in selected])

    @classmethod
    def combine(cls, models: Iterable[ForceConstants]) -> ForceConstants:
        """Combine disjoint orders using the first model's cluster space.

        Callers must ensure all coefficient vectors use that parameter layout.
        Empty input, duplicate orders and invalid coefficient lengths are rejected;
        cross-model geometry and basis compatibility are not checked.
        """
        items = tuple(models)
        if not items:
            raise ValueError("at least one force-constant model is required")
        space = items[0].cluster_space
        coefficients: dict[int, np.ndarray] = {}
        for model in items:
            overlap = coefficients.keys() & model.coefficients.keys()
            if overlap:
                raise ValueError(f"force-constant orders are duplicated: {tuple(sorted(overlap))}")
            coefficients.update(model.coefficients)
        return cls(space, coefficients)

    def save(self, file: str | os.PathLike[str]) -> Path:
        """Atomically save primitive geometry, bases, masses and coefficients as HDF5.

        file is a path-like target with any extension, conventionally .mlfcs.
        Return its absolute Path. Native format version 5 stores int64 and
        float64 arrays without executable object encoding. I/O errors propagate;
        a failed write leaves an existing target unchanged.
        """
        from mlfcs.force_constants import native as _native

        path = Path(file).resolve()
        started = perf_counter()
        logger.info("Native save started: path=%s orders=%s", path, self.orders)
        result = _native.save(self, path)
        logger.info("Native save complete: path=%s elapsed_s=%.2f", path, perf_counter() - started)
        return result

    @classmethod
    def load(cls, file: str | os.PathLike[str]) -> ForceConstants:
        """Load a validated native version-five HDF5 force-constant model.

        file is a path-like input. Restore physical coefficients and readonly
        primitive geometry, masses and bases without repeating orbit construction.
        Raise ValueError for unsupported formats or invalid stored arrays/layout;
        file-access errors propagate. Legacy pickle files are not supported.
        """
        from mlfcs.force_constants import native as _native

        path = Path(file).resolve()
        started = perf_counter()
        logger.info("Native load started: path=%s", path)
        model = _native.load(path)
        logger.info(
            "Native load complete: path=%s orders=%s elapsed_s=%.2f",
            path,
            model.orders,
            perf_counter() - started,
        )
        return model

    def write(
        self,
        file: str | os.PathLike[str],
        cluster_map: ClusterMap | None = None,
        *,
        format: Literal["phonopy", "phono3py", "shengbte"],
        order: int,
        storage: Literal["text", "hdf5"] | None = None,
        threshold: float = 1e-8,
    ) -> Path:
        """Overwrite one order in an external force-constant format and return its path.

        Parameters
        ----------
        file : path-like
            Target file; parent directories are created.
        cluster_map : ClusterMap, optional
            Matching supercell relation required for phonopy and phono3py.
            ShengBTE uses primitive lattice data and needs no map.
            Callers own map/model compatibility.
        format : {'phonopy', 'phono3py', 'shengbte'}
            phonopy supports FC2; phono3py FC3; ShengBTE FC3/FC4.
        order : int
            Explicitly present order to export.
        storage : {'text', 'hdf5'}, optional
            phonopy defaults to text and permits HDF5; phono3py requires HDF5;
            ShengBTE requires text.
        threshold : float, default 1e-8
            Nonnegative physical component cutoff; smaller absolute entries become zero.

        Notes
        -----
        The model is unchanged. Native save/load is separate from external formats.
        Invalid format/order/storage or missing required tensor information raises ValueError.
        """
        from mlfcs.force_constants.compact import CompactForceConstants

        return CompactForceConstants(self, cluster_map).write(
            file,
            format=format,
            order=order,
            storage=storage,
            threshold=threshold,
        )

    def enforce_rotation(
        self,
        *,
        born_huang: bool = True,
        huang: bool = False,
        rank_rtol: float | None = None,
    ) -> RotationResult:
        """Correct FC2 rotational moments without changing its acoustic residual.

        ``born_huang`` selects first-moment rotational conditions for atoms
        in force equilibrium. ``huang`` additionally selects zero-stress
        second-moment conditions and is disabled by default. ``rank_rtol``
        sets a relative singular-value cutoff between zero and one; when
        omitted, it is estimated from the actual symmetry mismatch of the
        geometry, with a machine-precision floor.

        Return a RotationResult containing a new model and residual/rank
        diagnostics. The correction minimizes change in physical FC2
        parameters within resolved directions; other orders are unchanged.
        The cluster space must have been initialized with ``asr=True``.
        Invalid selections, missing FC2 or an undefined pair length scale
        raise ValueError.
        """
        from mlfcs.force_constants.rotation import enforce_rotation

        return enforce_rotation(
            self,
            born_huang=born_huang,
            huang=huang,
            rank_rtol=rank_rtol,
        )


__all__ = ["ForceConstants"]
