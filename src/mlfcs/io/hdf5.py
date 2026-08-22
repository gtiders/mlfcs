from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from mlfcs.model import ForceConstants


def write_hdf5(target: str | Path, force_constants: ForceConstants) -> None:
    with h5py.File(target, "w") as handle:
        handle.attrs["format"] = "mlfcs-force-constants"
        handle.attrs["units"] = "eV/angstrom^order"
        handle.attrs["atom_order"] = "cell-major:z,y,x,primitive_atom"
        handle.create_dataset("cell", data=np.asarray(force_constants.supercell.cell))
        handle.create_dataset("positions", data=force_constants.supercell.get_positions())
        handle.create_dataset("numbers", data=force_constants.supercell.numbers)
        ordering = handle.create_group("ordering")
        for name in (
            "primitive_index",
            "cell_translation",
            "primitive_scaled_position",
        ):
            if name in force_constants.supercell.arrays:
                ordering.create_dataset(name, data=force_constants.supercell.arrays[name])
        group = handle.create_group("force_constants")
        for order, values in sorted(force_constants.arrays.items()):
            if order in force_constants.sparse:
                continue
            dataset = group.create_dataset(str(order), data=values, compression="gzip")
            dataset.attrs["representation"] = "dense"
            dataset.attrs["order"] = order
            dataset.attrs["unit"] = f"eV/angstrom^{order}"
        for order, values in sorted(force_constants.sparse.items()):
            order_group = group.create_group(str(order))
            order_group.attrs["representation"] = "sparse-cluster"
            order_group.attrs["order"] = order
            order_group.attrs["unit"] = f"eV/angstrom^{order}"
            order_group.attrs["dense_shape"] = values.dense_shape
            order_group.create_dataset("clusters", data=values.clusters, compression="gzip")
            order_group.create_dataset("tensors", data=values.tensors, compression="gzip")
        for key, value in force_constants.metadata.items():
            if isinstance(value, (str, int, float, bool, np.number)):
                handle.attrs[key] = value
