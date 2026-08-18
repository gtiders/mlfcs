from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from mlfcs.anharmonic.common.fc2 import expand_compact_fc2


def write_phonopy(
    target: str | Path,
    force_constants: np.ndarray,
    supercell: Atoms,
) -> None:
    """Write full second-order force constants in phonopy text format.

    The input uses mlfcs' compact ``(n_primitive, n_supercell, 3, 3)``
    representation. The file uses phonopy's full supercell representation
    while preserving the explicit reference-supercell atom order.
    """
    compact = np.asarray(force_constants)
    n_supercell = len(supercell)
    primitive = np.asarray(supercell.arrays["primitive_index"], dtype=np.int64)
    n_primitive = int(primitive.max()) + 1
    expected = (n_primitive, n_supercell, 3, 3)
    if compact.shape != expected:
        raise ValueError(f"compact FC2 must have shape {expected}, got {compact.shape}")

    full_internal = expand_compact_fc2(compact, supercell)

    # Atom axes retain the explicit reference-supercell order. Phonopy does
    # not require primitive-site grouping; its structure and FC2 labels only
    # have to use the same order.
    full = full_internal
    lines = [f"{n_supercell:4d} {n_supercell:4d}"]
    for first in range(n_supercell):
        for second in range(n_supercell):
            lines.append(f"{first + 1:d} {second + 1:d}")
            for vector in full[first, second]:
                lines.append(("%22.15f" * 3) % tuple(vector))
    Path(target).write_text("\n".join(lines))


__all__ = ["write_phonopy"]
