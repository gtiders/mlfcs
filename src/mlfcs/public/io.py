"""Public force-constant persistence and export API."""

from pathlib import Path

from mlfcs.ifc.model import ForceConstants
from mlfcs.io import write_force_constants


def read_hdf5(source: str | Path) -> ForceConstants:
    """Read native MLFCS HDF5 schema v3 force constants.

    Older native schemas are rejected because their atom-order semantics are
    not recoverable without guessing.
    """
    from mlfcs.io.hdf5 import read_hdf5 as _read_hdf5

    return _read_hdf5(source)


__all__ = ["read_hdf5", "write_force_constants"]
