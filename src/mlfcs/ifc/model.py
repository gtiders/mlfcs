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
    cutoff: float | int | None = -5
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
    """Sparse IFCs with both calculation and physical lattice labels.

    ``clusters`` is a reference-frame index view used by existing numerical
    kernels. ``sites`` and ``translation_representatives`` are the physical
    storage identity: ``Phi[k1,...,kn](0,R2,...,Rn)``.
    """

    order: int
    n_primitive: int
    n_supercell: int
    clusters: np.ndarray
    tensors: np.ndarray
    sites: np.ndarray | None = None
    translation_representatives: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.clusters = np.asarray(self.clusters, dtype=np.int32).reshape((-1, self.order))
        self.tensors = np.asarray(self.tensors, dtype=float).reshape((-1,) + (3,) * self.order)
        if len(self.clusters) != len(self.tensors):
            raise ValueError("sparse cluster and tensor counts differ")
        if (self.sites is None) != (self.translation_representatives is None):
            raise ValueError("sites and translation_representatives must be supplied together")
        if self.sites is not None:
            self.sites = np.asarray(self.sites, dtype=np.int32).reshape((-1, self.order))
            self.translation_representatives = np.asarray(
                self.translation_representatives, dtype=np.int32
            ).reshape((-1, self.order - 1, 3))
            if len(self.sites) != len(self.clusters) or len(
                self.translation_representatives
            ) != len(self.clusters):
                raise ValueError("lattice-labelled sparse IFC arrays have incompatible lengths")
            if np.any(self.sites < 0) or np.any(self.sites >= self.n_primitive):
                raise ValueError("sparse IFC primitive site label is out of range")

    @property
    def is_lattice_labelled(self) -> bool:
        return self.sites is not None

    @property
    def dense_shape(self) -> tuple[int, ...]:
        return (self.n_primitive,) + (self.n_supercell,) * (self.order - 1) + (3,) * self.order

    @property
    def dense_nbytes(self) -> int:
        return int(np.prod(self.dense_shape, dtype=np.int64)) * self.tensors.dtype.itemsize

    @property
    def support(self) -> np.ndarray:
        """Return the symmetry-closed atomic-cluster support."""
        result = np.zeros(self.dense_shape[: self.order], dtype=bool)
        if len(self.clusters):
            anchors = self.clusters[:, 0] if self.sites is None else self.sites[:, 0]
            result[(anchors, *self.clusters[:, 1:].T)] = True
        return result

    def to_dense(
        self,
        *,
        max_bytes: int | None = 2_000_000_000,
        primitive_index: np.ndarray | None = None,
    ) -> np.ndarray:
        """Materialize the compact tensor, warning when it exceeds the budget."""
        if max_bytes is not None and self.dense_nbytes > max_bytes:
            gib = self.dense_nbytes / 1024**3
            warnings.warn(
                f"dense order-{self.order} force constants require {gib:.2f} GiB; "
                "materialization will continue; sparse HDF5 output is safer",
                RuntimeWarning,
                stacklevel=2,
            )
        result = np.zeros(self.dense_shape, dtype=self.tensors.dtype)
        counts = np.zeros(self.dense_shape[: self.order], dtype=np.int16)
        for row, (cluster, tensor) in enumerate(zip(self.clusters, self.tensors, strict=True)):
            key = tuple(int(atom) for atom in cluster)
            if primitive_index is not None:
                key = (int(primitive_index[key[0]]), *key[1:])
            elif self.sites is not None:
                key = (int(self.sites[row, 0]), *key[1:])
            result[key] += tensor
            counts[key] += 1
        nonzero = counts > 0
        result[nonzero] /= counts[nonzero].reshape((-1,) + (1,) * self.order)
        return result


@dataclass(slots=True)
class ForceConstants:
    """Compact force constants and the supercell defining their indices."""

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
        primitive_index = self.supercell.arrays.get("primitive_index")
        if primitive_index is None and self.relation is not None:
            primitive_index = self.relation.primitive_index
        values = self.sparse[order].to_dense(max_bytes=max_bytes, primitive_index=primitive_index)
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


def replace_compact_fc2(
    base: ForceConstants,
    compact: np.ndarray,
    *,
    support: set[tuple[int, int, tuple[int, int, int]]] | None = None,
    metadata: dict[str, object] | None = None,
) -> ForceConstants:
    """Return a standard sparse FC2 object with ``compact`` values.

    ``base`` supplies the structure relation and lattice-labelled FC2 support.
    The optional support can extend that support with translation-labelled pairs
    required by a physical correction.  The returned object intentionally
    contains only FC2: a self-consistent effective harmonic result must not
    silently carry the input FC3 or FC4 terms.
    """
    if base.relation is None or 2 not in base.sparse:
        raise ValueError("FC2 replacement requires lattice-labelled sparse FC2 and StructureRelation")
    sparse_base = base.sparse[2]
    if sparse_base.sites is None or sparse_base.translation_representatives is None:
        raise ValueError("FC2 replacement requires lattice-labelled sparse FC2")
    relation = base.relation
    index = relation.index
    values = np.asarray(compact, dtype=float)
    expected = (sparse_base.n_primitive, sparse_base.n_supercell, 3, 3)
    if values.shape != expected:
        raise ValueError(f"compact FC2 must have shape {expected}, got {values.shape}")

    rows: list[tuple[int, int]] = [tuple(map(int, cluster)) for cluster in sparse_base.clusters]
    row_keys = {
        (int(sites[0]), int(sites[1]), tuple(map(int, translations[0])))
        for sites, translations in zip(
            sparse_base.sites, sparse_base.translation_representatives, strict=True
        )
    }
    if support is not None:
        for site, other, translation in sorted(support):
            key = (site, other, translation)
            if key in row_keys:
                continue
            rows.append((index.representative(site), index.atom(other, translation)))
            row_keys.add(key)

    clusters = np.asarray(rows, dtype=np.int32).reshape((-1, 2))
    primitive_index = np.asarray(relation.primitive_index, dtype=np.int64)
    tensors = np.asarray(
        [values[int(primitive_index[first]), int(second)] for first, second in clusters], dtype=float
    )
    sites = index.primitive[clusters]
    translations = np.asarray(
        [
            [index.canonical_translation(index.translations[second] - index.translations[first])]
            for first, second in clusters
        ],
        dtype=np.int32,
    )
    sparse = SparseOrderForceConstants(
        2,
        sparse_base.n_primitive,
        sparse_base.n_supercell,
        clusters,
        tensors,
        sites,
        translations,
    )
    result_metadata = dict(base.metadata)
    if metadata is not None:
        result_metadata.update(metadata)
    return ForceConstants({}, relation.reference.copy(), result_metadata, {2: sparse}, relation)


__all__ = ["ForceConstants", "RunConfig", "SparseOrderForceConstants", "replace_compact_fc2"]
