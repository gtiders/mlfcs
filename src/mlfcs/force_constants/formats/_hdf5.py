"""Shared phonon HDF5 writer."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np

from mlfcs.force_constants.export import compact, primitive_to_supercell, translated_atoms
from mlfcs.force_constants.model import ForceConstants
from mlfcs.supercell import ClusterMap


def write_hdf5(
    path: Path,
    model: ForceConstants,
    mapping: ClusterMap,
    *,
    order: int,
    threshold: float,
) -> None:
    """Write full-supercell FC2 or FC3 in phonon HDF5 conventions."""
    values = compact(model, mapping, order, threshold=threshold)
    size = len(mapping.supercell.numbers)
    shape = (size,) * order + (3,) * order
    name = "force_constants" if order == 2 else "fc3"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            name,
            shape=shape,
            dtype=np.float64,
            chunks=(1,) + shape[1:],
            compression="gzip",
            compression_opts=4,
        )
        for first in range(size):
            tails = translated_atoms(mapping, first)
            primitive = int(mapping.supercell.sites[first])
            if order == 2:
                dataset[first] = values[primitive, tails]
            else:
                dataset[first] = values[primitive][np.ix_(tails, tails)]
        handle.create_dataset("p2s_map", data=primitive_to_supercell(mapping))
        try:
            release = version("mlfcs")
        except PackageNotFoundError:
            release = "unknown"
        handle.create_dataset("version", data=np.bytes_(f"mlfcs {release}"))
        if order == 2:
            handle.create_dataset("physical_unit", data=np.asarray([b"eV/angstrom^2"]))
        handle.attrs["mlfcs_threshold"] = threshold


__all__ = ["write_hdf5"]
