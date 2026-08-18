from __future__ import annotations

import numpy as np
from ase import Atoms

from mlfcs.structure.geometry import PeriodicIndex


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


__all__ = ["compact_fc2", "expand_compact_fc2"]
