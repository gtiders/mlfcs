"""External force-constant output owned by :class:`ForceConstants`."""

from __future__ import annotations

import os
from pathlib import Path

from mlfcs.force_constants.export import DEFAULT_THRESHOLD, threshold_value, validate
from mlfcs.force_constants.formats.phono3py import write_phono3py
from mlfcs.force_constants.formats.phonopy import write_phonopy
from mlfcs.force_constants.formats.shengbte import write_shengbte
from mlfcs.force_constants.formats.tdep import write_tdep
from mlfcs.force_constants.model import ForceConstants
from mlfcs.supercell import ClusterMap


def write(
    model: ForceConstants,
    file: str | os.PathLike[str],
    mapping: ClusterMap | None,
    *,
    format: str,
    order: int,
    storage: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Path:
    """Write one explicitly selected order in one supported external format."""
    path = Path(file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    threshold = threshold_value(threshold)
    if format == "phonopy":
        if order != 2:
            raise ValueError("phonopy output supports only order 2")
        selected_storage = "text" if storage is None else storage
        if selected_storage not in {"text", "hdf5"}:
            raise ValueError("phonopy storage must be 'text' or 'hdf5'")
        validate(model, mapping, order)
        write_phonopy(path, model, mapping, storage=selected_storage, threshold=threshold)
    elif format == "phono3py":
        if order != 3:
            raise ValueError("phono3py output supports only order 3")
        if storage not in {None, "hdf5"}:
            raise ValueError("phono3py output uses HDF5 storage")
        validate(model, mapping, order)
        write_phono3py(path, model, mapping, threshold=threshold)
    elif format == "shengbte":
        if order not in {3, 4}:
            raise ValueError("ShengBTE output supports only orders 3 and 4")
        if storage not in {None, "text"}:
            raise ValueError("ShengBTE output uses text storage")
        validate(model, mapping, order)
        write_shengbte(path, model, mapping, order=order, threshold=threshold)
    elif format == "tdep":
        if order not in {2, 3, 4}:
            raise ValueError("TDEP output supports only orders 2, 3 and 4")
        if storage not in {None, "text"}:
            raise ValueError("TDEP output uses text storage")
        if order not in model.coefficients:
            raise ValueError(f"force constants do not contain order {order}")
        if mapping is not None:
            validate(model, mapping, order)
        write_tdep(path, model, order=order, threshold=threshold)
    else:
        raise ValueError(
            f"unsupported force-constant format {format!r}; supported formats are "
            "phonopy, phono3py, shengbte, and tdep"
        )
    return path


__all__ = ["write"]
