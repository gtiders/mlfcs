"""Primitive force constants in cluster-space coordinates."""

from __future__ import annotations

import hashlib
import os
import pickle
import struct
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np

from mlfcs.cluster_space import ClusterSpace
from mlfcs.core.log import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mlfcs.force_constants.asr import ASRResult
    from mlfcs.force_constants.rotation import RotationResult
    from mlfcs.mapping import ClusterMap

_FORMAT = "mlfcs.force_constants"
_VERSION = 3
_HEADER = b"MLFCS\x00\x03\n"


def _digest(space: ClusterSpace, coefficients: Mapping[int, np.ndarray]) -> str:
    """Hash primitive model identity and ordered physical coefficients in fixed byte order."""
    digest = hashlib.sha256()
    digest.update(space.fingerprint.encode("ascii"))
    for order in sorted(coefficients):
        values = np.ascontiguousarray(coefficients[order], dtype=np.float64)
        digest.update(struct.pack(">q", order))
        digest.update(struct.pack(">q", len(values)))
        digest.update(values.astype(values.dtype.newbyteorder(">"), copy=False).tobytes())
    return digest.hexdigest()


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
    not arbitrary exact lattice-kernel generator coefficients.

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

    def __reduce__(self):
        """Serialize the shared model schema and physical coefficient vectors for validation on
        load.
        """
        return type(self), (self.cluster_space, dict(self.coefficients))

    @property
    def orders(self) -> tuple[int, ...]:
        """Explicitly present tensor orders in ascending order."""
        return tuple(sorted(self.coefficients))

    @property
    def fingerprint(self) -> str:
        """Stable hash of the cluster-space identity and physical coefficient contents."""
        return _digest(self.cluster_space, self.coefficients)

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
        """Combine disjoint orders defined on the same cluster space."""
        items = tuple(models)
        if not items:
            raise ValueError("at least one force-constant model is required")
        space = items[0].cluster_space
        coefficients: dict[int, np.ndarray] = {}
        for model in items:
            if model.cluster_space.fingerprint != space.fingerprint:
                raise ValueError("force constants use different cluster spaces")
            overlap = coefficients.keys() & model.coefficients.keys()
            if overlap:
                raise ValueError(f"force-constant orders are duplicated: {tuple(sorted(overlap))}")
            coefficients.update(model.coefficients)
        return cls(space, coefficients)

    def save(self, file: str | os.PathLike[str]) -> Path:
        """Atomically write the native trusted-pickle representation."""
        path = Path(file).resolve()
        started = perf_counter()
        logger.info("Native save started: path=%s orders=%s", path, self.orders)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _FORMAT,
            "version": _VERSION,
            "fingerprint": self.fingerprint,
            "cluster_space": self.cluster_space._state(),
            "coefficients": dict(self.coefficients),
        }
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
                temporary = handle.name
                handle.write(_HEADER)
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        logger.info("Native save complete: path=%s elapsed_s=%.2f", path, perf_counter() - started)
        return path

    @classmethod
    def load(cls, file: str | os.PathLike[str]) -> ForceConstants:
        """Read a native force-constant file from a trusted source."""
        path = Path(file).resolve()
        started = perf_counter()
        logger.info("Native load started: path=%s", path)
        with path.open("rb") as handle:
            if handle.read(len(_HEADER)) != _HEADER:
                raise ValueError("unsupported force-constant file version; version 3 is required")
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
            raise ValueError(f"{path} is not an MLFCS force-constant file")
        if payload.get("version") != _VERSION:
            raise ValueError(
                f"unsupported force-constant version {payload.get('version')}; this release reads version {_VERSION}"
            )
        from mlfcs.cluster_space.models import _restore_cluster_space

        space = _restore_cluster_space(payload["cluster_space"])
        model = cls(space, payload.get("coefficients"))
        if payload.get("fingerprint") != model.fingerprint:
            raise ValueError("force-constant fingerprint does not match its contents")
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
        format: Literal["phonopy", "phono3py", "shengbte", "tdep"],
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
            Matching supercell relation required for phonopy, phono3py and ShengBTE.
            TDEP uses primitive lattice data; an optional map is identity-checked.
        format : {'phonopy', 'phono3py', 'shengbte', 'tdep'}
            phonopy supports FC2; phono3py FC3; ShengBTE FC3/FC4; TDEP FC2/FC3/FC4.
        order : int
            Explicitly present order to export.
        storage : {'text', 'hdf5'}, optional
            phonopy defaults to text and permits HDF5; phono3py requires HDF5;
            ShengBTE and TDEP require text.
        threshold : float, default 1e-8
            Nonnegative physical component cutoff; smaller absolute entries become zero.

        Notes
        -----
        The model is unchanged. Native save/load is separate from external formats.
        Invalid format/order/storage or mismatched model relation raises ValueError.
        """
        from mlfcs.force_constants.io import write

        return write(
            self,
            file,
            cluster_map,
            format=format,
            order=order,
            storage=storage,
            threshold=threshold,
        )

    def enforce_asr(
        self,
        *,
        orders: Iterable[int] | None = None,
        rtol: float = 1e-10,
    ) -> ASRResult:
        """Return the shared post-processing projection onto translational invariance."""
        from mlfcs.force_constants.asr import enforce_asr

        return enforce_asr(self, orders=orders, rtol=rtol)

    def enforce_rotation(
        self,
        *,
        born_huang: bool = True,
        huang: bool = False,
        rank_rtol: float | None = None,
    ) -> RotationResult:
        """Apply resolvable rotational conditions without changing the existing ASR."""
        from mlfcs.force_constants.rotation import enforce_rotation

        return enforce_rotation(
            self,
            born_huang=born_huang,
            huang=huang,
            rank_rtol=rank_rtol,
        )


__all__ = ["ForceConstants"]
