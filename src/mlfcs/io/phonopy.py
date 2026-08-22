from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms


def write_phonopy(
    target: str | Path,
    force_constants: np.ndarray,
    supercell: Atoms,
) -> None:
    """Write full second-order force constants in phonopy text format.

    The input uses mlfcs' compact ``(n_primitive, n_supercell, 3, 3)``
    representation. The file uses phonopy's full supercell representation
    and primitive-atom-grouped supercell ordering.
    """
    compact = np.asarray(force_constants)
    primitive = np.asarray(supercell.arrays["primitive_index"], dtype=np.int64)
    translations = np.asarray(supercell.arrays["cell_translation"], dtype=np.int64)
    n_supercell = len(supercell)
    n_primitive = int(primitive.max()) + 1
    expected = (n_primitive, n_supercell, 3, 3)
    if compact.shape != expected:
        raise ValueError(f"compact FC2 must have shape {expected}, got {compact.shape}")

    repeats = translations.max(axis=0) + 1
    atom_by_key = {
        (int(p), *(int(value) for value in translation)): atom
        for atom, (p, translation) in enumerate(zip(primitive, translations, strict=True))
    }
    full_internal = np.empty((n_supercell, n_supercell, 3, 3), dtype=compact.dtype)
    for first in range(n_supercell):
        relative = np.mod(translations - translations[first], repeats)
        second = np.fromiter(
            (
                atom_by_key[(int(p), *(int(value) for value in translation))]
                for p, translation in zip(primitive, relative, strict=True)
            ),
            dtype=np.int64,
            count=n_supercell,
        )
        full_internal[first] = compact[int(primitive[first]), second]

    # Phonopy groups all translated images of each primitive atom together.
    grouped = np.lexsort((translations[:, 0], translations[:, 1], translations[:, 2], primitive))
    full = full_internal[grouped][:, grouped]
    lines = [f"{n_supercell:4d} {n_supercell:4d}"]
    for first in range(n_supercell):
        for second in range(n_supercell):
            lines.append(f"{first + 1:d} {second + 1:d}")
            for vector in full[first, second]:
                lines.append(("%22.15f" * 3) % tuple(vector))
    Path(target).write_text("\n".join(lines))


__all__ = ["write_phonopy"]
