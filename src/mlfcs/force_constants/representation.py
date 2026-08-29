from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SparseOrderForceConstants:
    """Canonical exact primitive-lattice IFC rows."""

    order: int
    sites: np.ndarray
    translations: np.ndarray
    tensors: np.ndarray

    def __post_init__(self) -> None:
        self.sites = np.asarray(self.sites, dtype=np.int32).reshape((-1, self.order))
        self.translations = np.asarray(self.translations, dtype=np.int32).reshape(
            (-1, self.order - 1, 3)
        )
        self.tensors = np.asarray(self.tensors, dtype=float).reshape((-1,) + (3,) * self.order)
        if len(self.sites) != len(self.translations) or len(self.sites) != len(self.tensors):
            raise ValueError("exact sparse IFC arrays have incompatible row counts")
        if np.any(self.sites < 0):
            raise ValueError("primitive site labels must be non-negative")


@dataclass(slots=True)
class ForceConstants:
    """Primitive exact-R IFCs with an optional realized target view.

    ``sparse`` is the canonical physical model. ``supercell`` and ``relation``
    identify the current finite view used only by materialization and writers;
    they are not part of an interaction's identity.
    """

    arrays: dict[int, np.ndarray]
    supercell: Atoms
    metadata: dict[str, object] = field(default_factory=dict)
    sparse: dict[int, SparseOrderForceConstants] = field(default_factory=dict)
    relation: object | None = None
    _export_view_cache: dict[tuple[object, ...], object] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(sorted(self.arrays.keys() | self.sparse.keys()))

    def materialize(self, order: int, *, max_bytes: int | None = 2_000_000_000) -> np.ndarray:
        if order in self.arrays:
            return self.arrays[order]
        if self.relation is None:
            raise ValueError("materialization requires an explicit target structure relation")
        sparse = self.sparse[order]
        relation = self.relation
        shape = (len(relation.primitive),) + (len(relation.reference),) * (order - 1)
        shape += (3,) * order
        nbytes = int(np.prod(shape, dtype=np.int64)) * sparse.tensors.dtype.itemsize
        if max_bytes is not None and nbytes > max_bytes:
            logger.warning(
                "Dense order-%d force constants require %.2f GiB; materialization will "
                "continue; sparse HDF5 output is safer",
                order,
                nbytes / 1024**3,
            )
        values = np.zeros(shape, dtype=sparse.tensors.dtype)
        for sites, translations, tensor in zip(
            sparse.sites, sparse.translations, sparse.tensors, strict=True
        ):
            atoms = tuple(
                relation.index.atom(int(site), translation)
                for site, translation in zip(sites[1:], translations, strict=True)
            )
            values[(int(sites[0]), *atoms)] += tensor
        self.arrays[order] = values
        return values


__all__ = [
    "ForceConstants",
    "SparseOrderForceConstants",
]
