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
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np

from mlfcs.cluster_space import ClusterSpace

if TYPE_CHECKING:
    from mlfcs.force_constants.asr import ASRResult
    from mlfcs.force_constants.rotation import RotationResult
    from mlfcs.supercell import ClusterMap

_FORMAT = "mlfcs.force_constants"
_VERSION = 1


def _digest(space: ClusterSpace, coefficients: Mapping[int, np.ndarray]) -> str:
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
    """One primitive-cell force-constant model.

    Coefficients use the canonical parameter layout of ``space``. Only the
    explicitly present orders belong to the model; an absent order is never
    interpreted as zero. Supercell expansion is a derived operation and is
    deliberately not stored here.
    """

    space: ClusterSpace
    coefficients: Mapping[int, np.ndarray]

    def __post_init__(self) -> None:
        if not isinstance(self.space, ClusterSpace):
            raise TypeError("space must be a ClusterSpace")
        values: dict[int, np.ndarray] = {}
        for key, source in self.coefficients.items():
            order = int(key)
            if order != key:
                raise TypeError("force-constant orders must be integers")
            block = self.space.block(order)
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
        return type(self), (self.space, dict(self.coefficients))

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(sorted(self.coefficients))

    @property
    def fingerprint(self) -> str:
        return _digest(self.space, self.coefficients)

    def parameters(self, orders: Iterable[int] | None = None) -> np.ndarray:
        """Return a packed copy in ascending-order cluster-space layout."""
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
        space = items[0].space
        coefficients: dict[int, np.ndarray] = {}
        for model in items:
            if model.space.fingerprint != space.fingerprint:
                raise ValueError("force constants use different cluster spaces")
            overlap = coefficients.keys() & model.coefficients.keys()
            if overlap:
                raise ValueError(f"force-constant orders are duplicated: {tuple(sorted(overlap))}")
            coefficients.update(model.coefficients)
        return cls(space, coefficients)

    def save(self, file: str | os.PathLike[str]) -> Path:
        """Atomically write the native trusted-pickle representation."""
        path = Path(file).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _FORMAT,
            "version": _VERSION,
            "fingerprint": self.fingerprint,
            "space": self.space,
            "coefficients": dict(self.coefficients),
        }
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
                temporary = handle.name
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return path

    @classmethod
    def load(cls, file: str | os.PathLike[str]) -> ForceConstants:
        """Read a native force-constant file from a trusted source."""
        path = Path(file).resolve()
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
            raise ValueError(f"{path} is not an MLFCS force-constant file")
        if payload.get("version") != _VERSION:
            raise ValueError(
                f"unsupported force-constant version {payload.get('version')}; "
                f"this release reads version {_VERSION}"
            )
        model = cls(payload.get("space"), payload.get("coefficients"))
        if payload.get("fingerprint") != model.fingerprint:
            raise ValueError("force-constant fingerprint does not match its contents")
        return model

    def write(
        self,
        file: str | os.PathLike[str],
        mapping: ClusterMap | None = None,
        *,
        format: Literal["phonopy", "phono3py", "shengbte", "tdep"],
        order: int,
        storage: Literal["text", "hdf5"] | None = None,
        threshold: float = 1e-8,
    ) -> Path:
        """Write one order in a supported external force-constant format.

        ``mapping`` selects the target supercell for phonopy, phono3py and
        ShengBTE. TDEP instead uses primitive-lattice clusters and needs no
        mapping; one supplied for TDEP is checked for model identity. Small
        components are set to zero after primitive-lattice expansion.
        """
        from mlfcs.force_constants.io import write

        return write(
            self,
            file,
            mapping,
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
