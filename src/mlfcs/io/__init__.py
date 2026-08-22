from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlfcs.model import ForceConstants

Writer = Callable[..., None]


def write_force_constants(
    force_constants: ForceConstants,
    target: str | Path,
    *,
    format: str,
    order: int | None = None,
    compatibility: str | None = None,
) -> None:
    normalized = format.casefold().replace("-", "_")
    if compatibility is not None and normalized != "shengbte":
        raise ValueError("compatibility is available only for ShengBTE output")
    if normalized == "hdf5":
        from mlfcs.io.hdf5 import write_hdf5

        write_hdf5(target, force_constants)
        return
    if normalized in {"numpy", "npz"}:
        from mlfcs.io.numpy import write_numpy

        write_numpy(target, force_constants)
        return
    if normalized == "phonopy":
        from mlfcs.io.phonopy import write_phonopy

        selected_order = order if order is not None else max(force_constants.orders)
        if selected_order != 2:
            raise ValueError("phonopy text output is available only for order 2")
        write_phonopy(
            target,
            force_constants.materialize(2),
            force_constants.supercell,
        )
        return
    if normalized in {"phonopy_hdf5", "phonopy_h5"}:
        from mlfcs.io.phonon_hdf5 import write_phonon_hdf5

        selected_order = order if order is not None else max(force_constants.orders)
        if selected_order != 2:
            raise ValueError("phonopy HDF5 output is available only for order 2")
        write_phonon_hdf5(target, force_constants, order=2)
        return
    if normalized in {"phono3py_hdf5", "phono3py_h5"}:
        from mlfcs.io.phonon_hdf5 import write_phonon_hdf5

        selected_order = order if order is not None else max(force_constants.orders)
        if selected_order != 3:
            raise ValueError("phono3py HDF5 output is available only for order 3")
        write_phonon_hdf5(target, force_constants, order=3)
        return
    if normalized == "shengbte":
        from mlfcs.io.shengbte import write_shengbte

        selected_order = order if order is not None else max(force_constants.orders)
        cutoff = force_constants.metadata.get("cutoff_angstrom")
        if cutoff is None:
            raise ValueError("cutoff_angstrom metadata is required for ShengBTE output")
        sparse = force_constants.sparse.get(selected_order)
        values = force_constants.materialize(selected_order)
        write_shengbte(
            target,
            values,
            force_constants.supercell,
            cutoff=float(cutoff),
            support=None if sparse is None else sparse.support,
            compatibility=compatibility,
        )
        return
    raise ValueError(f"unknown force-constant format: {format!r}")


__all__ = ["write_force_constants"]
