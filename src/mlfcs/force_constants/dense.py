from __future__ import annotations

import numpy as np
from ase import Atoms

from mlfcs.force_constants.data import ForceConstants, SparseOrderForceConstants
from mlfcs.structure.supercell_mapping import PeriodicIndex


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
    """Return an FC2-only result from exact primitive-lattice tensors."""
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


def expand_compact_fc2(compact: np.ndarray, supercell: Atoms) -> np.ndarray:
    """Expand translation-reduced FC2 to the reference full-supercell order."""
    values = np.asarray(compact, dtype=float)
    index = _ordering(supercell)
    primitive, translations = index.primitive, index.translations
    n_supercell = len(supercell)
    n_primitive = int(primitive.max()) + 1
    expected = (n_primitive, n_supercell, 3, 3)
    if values.shape != expected:
        raise ValueError(f"compact FC2 must have shape {expected}, got {values.shape}")

    full = np.empty((n_supercell, n_supercell, 3, 3), dtype=values.dtype)
    for first in range(n_supercell):
        relative = translations - translations[first]
        second = np.fromiter(
            (
                index.atom(int(p), translation)
                for p, translation in zip(primitive, relative, strict=True)
            ),
            dtype=np.int64,
            count=n_supercell,
        )
        full[first] = values[int(primitive[first]), second]
    return full


def compact_fc2(full: np.ndarray, supercell: Atoms) -> np.ndarray:
    """Reduce a translation-covariant full FC2 in reference atom order."""
    values = np.asarray(full, dtype=float)
    n_supercell = len(supercell)
    expected = (n_supercell, n_supercell, 3, 3)
    if values.shape != expected:
        raise ValueError(f"full FC2 must have shape {expected}, got {values.shape}")
    primitive = np.asarray(supercell.arrays["primitive_index"], dtype=np.int64)
    n_primitive = int(primitive.max()) + 1
    representatives = [int(np.flatnonzero(primitive == atom)[0]) for atom in range(n_primitive)]
    return np.asarray(values[representatives]).copy()


def _ordering(supercell: Atoms) -> PeriodicIndex:
    try:
        primitive = np.asarray(supercell.arrays["primitive_index"], dtype=np.int64)
        translations = np.asarray(supercell.arrays["cell_translation"], dtype=np.int64)
    except KeyError as error:
        raise ValueError("supercell is missing MLFCS atom-order metadata") from error
    matrix = supercell.info.get("mlfcs_supercell_matrix")
    if matrix is None:
        raise ValueError("supercell is missing the MLFCS supercell-matrix metadata")
    return PeriodicIndex(primitive, translations, np.asarray(matrix, dtype=np.int32))


__all__ = [
    "compact_fc2",
    "expand_compact_fc2",
    "lattice_fc2",
    "replace_lattice_fc2",
]
