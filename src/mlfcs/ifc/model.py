from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration in ASE units (angstrom and eV)."""

    order: int
    supercell: object = (2, 2, 2)
    cutoff: float | int = -5
    max_body_order: int | None = None
    displacement: float = 0.01
    symprec: float = 1e-5

    def __post_init__(self) -> None:
        if self.order < 2:
            raise ValueError("order must be at least 2")
        if self.cutoff == 0:
            raise ValueError("cutoff cannot be zero")
        if self.max_body_order is not None and not 1 <= self.max_body_order <= self.order:
            raise ValueError("max_body_order must be between 1 and order")
        if self.displacement <= 0:
            raise ValueError("displacement must be positive")


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
            warnings.warn(
                f"dense order-{order} force constants require {nbytes / 1024**3:.2f} GiB; "
                "materialization will continue; sparse HDF5 output is safer",
                RuntimeWarning,
                stacklevel=2,
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

    def write(
        self,
        target: str | Path,
        *,
        format: str,
        order: int | None = None,
        primitive: Atoms | None = None,
        supercell: Atoms | None = None,
    ) -> None:
        from mlfcs.io import write_force_constants

        write_force_constants(
            self,
            target,
            format=format,
            order=order,
            primitive=primitive,
            supercell=supercell,
        )

    def realize(
        self,
        reference: Atoms,
        *,
        primitive: Atoms | None = None,
    ) -> ForceConstants:
        """Return these exact lattice-labelled IFCs in ``reference``.

        The target may contain a different number of primitive cells.  Exact
        real-space interactions that fold onto one finite cluster are summed
        when a dense target view is materialized.
        """
        from mlfcs.io.export import build_export_view

        return build_export_view(self, primitive=primitive, supercell=reference).force_constants

    def enforce_rotational_sum_rules(
        self,
        *,
        born_huang: bool = False,
        huang: bool = False,
        strength: float = 1.0,
        tolerance: float = 1e-8,
    ):
        """Return an FC2-only Born--Huang/Huang rotationally constrained result.

        This is an explicit postprocessing operation.  It leaves every order
        other than FC2 unchanged, including FC3 and FC4 from a Wick fit.
        """
        from mlfcs.constraints.rotational_sum_rules import enforce_rotational_sum_rules

        return enforce_rotational_sum_rules(
            self,
            born_huang=born_huang,
            huang=huang,
            strength=strength,
            tolerance=tolerance,
        )


def lattice_fc2(
    force_constants: ForceConstants,
) -> dict[tuple[int, int, tuple[int, int, int]], np.ndarray]:
    """Return FC2 tensors keyed by primitive sites and exact translation."""
    if 2 not in force_constants.sparse:
        raise ValueError("force constants do not contain FC2")
    sparse = force_constants.sparse[2]
    result: dict[tuple[int, int, tuple[int, int, int]], np.ndarray] = {}
    for sites, translations, tensor in zip(
        sparse.sites, sparse.translations, sparse.tensors, strict=True
    ):
        key = (int(sites[0]), int(sites[1]), tuple(map(int, translations[0])))
        result[key] = result.get(key, 0.0) + np.asarray(tensor, dtype=float)
    return result


def replace_lattice_fc2(
    base: ForceConstants,
    values: dict[tuple[int, int, tuple[int, int, int]], np.ndarray],
    *,
    metadata: dict[str, object] | None = None,
) -> ForceConstants:
    """Return an FC2-only result from exact primitive-lattice tensors.

    Several exact translations may fold to the same finite pair; they remain
    separate sparse rows and are summed only by materialization.
    """
    if base.relation is None:
        raise ValueError("FC2 replacement requires an explicit structure relation")
    relation = base.relation
    keys = sorted(values)
    sites = np.asarray([[first, second] for first, second, _ in keys], dtype=np.int32)
    translations = np.asarray([[translation] for _, _, translation in keys], dtype=np.int32)
    tensors = np.asarray([values[key] for key in keys], dtype=float).reshape((-1, 3, 3))
    sparse = SparseOrderForceConstants(2, sites, translations, tensors)
    result_metadata = dict(base.metadata)
    if metadata is not None:
        result_metadata.update(metadata)
    return ForceConstants({}, relation.reference.copy(), result_metadata, {2: sparse}, relation)


__all__ = [
    "ForceConstants",
    "RunConfig",
    "SparseOrderForceConstants",
    "lattice_fc2",
    "replace_lattice_fc2",
]
