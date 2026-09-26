"""Phono3py FC3 writer."""

from __future__ import annotations

from pathlib import Path

from mlfcs.force_constants.export import compact
from mlfcs.force_constants.formats._hdf5 import write_hdf5
from mlfcs.force_constants.model import ForceConstants
from mlfcs.supercell import ClusterMap


def write_phono3py(
    path: Path,
    model: ForceConstants,
    mapping: ClusterMap,
    *,
    threshold: float,
) -> None:
    """Write full-supercell FC3 in phono3py HDF5 convention."""
    values = compact(model, mapping, 3, threshold=threshold)
    write_hdf5(path, values, mapping, order=3, threshold=threshold)


__all__ = ["write_phono3py"]
