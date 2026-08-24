from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ase import Atoms

if TYPE_CHECKING:
    from mlfcs.force_constants.representation import ForceConstants

Writer = Callable[..., None]


def write_force_constants(
    force_constants: ForceConstants,
    target: str | Path,
    *,
    format: str,
    order: int | None = None,
    primitive: Atoms | None = None,
    supercell: Atoms | None = None,
) -> None:
    """Write force constants in a named external or native format.

    ``primitive`` and ``supercell`` may only describe an exactly equivalent
    representation; writers receive the resulting validated export view.
    """
    from mlfcs.force_constants.realization import build_export_view

    view = build_export_view(force_constants, primitive=primitive, supercell=supercell)
    force_constants = view.force_constants
    normalized = format.casefold().replace("-", "_")
    if normalized == "hdf5":
        from mlfcs.io.hdf5 import write_hdf5

        write_hdf5(target, force_constants)
        return
    if normalized in {"alamode", "alamode_xml", "fcsxml"}:
        from mlfcs.io.alamode import AlamodeMirrorImageError, reduced_export_view, write_alamode

        selected_orders = (
            tuple(value for value in force_constants.orders if value in {2, 3, 4})
            if order is None
            else (order,)
        )
        try:
            write_alamode(target, force_constants, orders=selected_orders)
        except AlamodeMirrorImageError as original_error:
            try:
                reduced = reduced_export_view(force_constants)
            except ValueError:
                raise original_error
            write_alamode(target, reduced.force_constants, orders=selected_orders)
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
        sparse = force_constants.sparse.get(selected_order)
        if sparse is None:
            raise ValueError("ShengBTE output requires lattice-labelled sparse force constants")
        write_shengbte(
            target,
            sparse,
            force_constants.supercell,
        )
        return
    raise ValueError(f"unknown force-constant format: {format!r}")


__all__ = ["write_force_constants"]
