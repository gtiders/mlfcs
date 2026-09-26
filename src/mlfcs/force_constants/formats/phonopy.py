"""Phonopy FC2 output from a folded primitive-first array."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from mlfcs.core.log_error import get_logger
from mlfcs.force_constants.export import DEFAULT_THRESHOLD, clean, translated_atoms
from mlfcs.force_constants.formats._hdf5 import write_hdf5
from mlfcs.supercell import ClusterMap

logger = get_logger(__name__)


def write_phonopy(
    file: str | os.PathLike[str],
    values: np.ndarray,
    mapping: ClusterMap,
    *,
    format: Literal["phonopy_text", "phonopy_hdf5"],
    threshold: float = DEFAULT_THRESHOLD,
) -> Path:
    """Write folded FC2 in the explicit supercell order of ``mapping``.

    ``values`` has shape ``(n_primitive, n_supercell, 3, 3)``. Its provenance
    and correspondence to ``mapping`` are the caller's responsibility.
    """
    if format not in {"phonopy_text", "phonopy_hdf5"}:
        raise ValueError("format must be 'phonopy_text' or 'phonopy_hdf5'")
    if not isinstance(mapping, ClusterMap):
        raise TypeError("mapping must be a ClusterMap")
    size = len(mapping.supercell.numbers)
    expected = (mapping.space.primitive.size, size, 3, 3)
    if np.shape(values) != expected:
        raise ValueError(f"phonopy FC2 must have shape {expected}, got {np.shape(values)}")
    cleaned = clean(values, threshold)
    path = Path(file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if format == "phonopy_hdf5":
        write_hdf5(path, cleaned, mapping, order=2, threshold=threshold)
    else:
        with path.open("w", encoding="ascii") as handle:
            handle.write(f"{size:4d} {size:4d}\n")
            for first in range(size):
                tails = translated_atoms(mapping, first)
                primitive = int(mapping.supercell.sites[first])
                for second in range(size):
                    handle.write(f"{first + 1:d} {second + 1:d}\n")
                    for row in cleaned[primitive, tails[second]]:
                        handle.write(("%22.15f" * 3) % tuple(row) + "\n")
    logger.info(
        "Exported FC2: format=%s threshold=%.6g file=%s elapsed=%.2f s",
        format,
        threshold,
        path,
        perf_counter() - started,
    )
    return path


__all__ = ["write_phonopy"]
