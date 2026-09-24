"""Phonopy FC2 writers."""

from __future__ import annotations

from pathlib import Path

from mlfcs.force_constants.export import compact, full_fc2
from mlfcs.force_constants.formats._hdf5 import write_hdf5
from mlfcs.force_constants.model import ForceConstants
from mlfcs.supercell import ClusterMap


def write_phonopy(
    path: Path,
    model: ForceConstants,
    mapping: ClusterMap,
    *,
    storage: str,
    threshold: float,
) -> None:
    """Write FC2 as phonopy text or HDF5 in explicit supercell order."""
    if storage == "hdf5":
        write_hdf5(path, model, mapping, order=2, threshold=threshold)
        return
    values = full_fc2(compact(model, mapping, 2, threshold=threshold), mapping)
    size = len(mapping.supercell.numbers)
    lines = [f"{size:4d} {size:4d}"]
    for first in range(size):
        for second in range(size):
            lines.append(f"{first + 1:d} {second + 1:d}")
            lines.extend(("%22.15f" * 3) % tuple(row) for row in values[first, second])
    path.write_text("\n".join(lines) + "\n")


__all__ = ["write_phonopy"]
