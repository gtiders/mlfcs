"""External force-constant output owned by :class:`ForceConstants`."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from mlfcs.core.log_error import get_logger
from mlfcs.force_constants.export import DEFAULT_THRESHOLD, threshold_value, validate
from mlfcs.force_constants.formats.phono3py import write_phono3py
from mlfcs.force_constants.formats.phonopy import write_phonopy
from mlfcs.force_constants.formats.shengbte import write_shengbte
from mlfcs.force_constants.formats.tdep import write_tdep
from mlfcs.force_constants.model import ForceConstants
from mlfcs.supercell import ClusterMap

logger = get_logger(__name__)


def write(
    model: ForceConstants,
    file: str | os.PathLike[str],
    mapping: ClusterMap | None,
    *,
    format: str,
    order: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> Path:
    """Write one explicitly selected order in one supported external format."""
    threshold = threshold_value(threshold)
    if format in {"phonopy_text", "phonopy_hdf5"}:
        if order != 2:
            raise ValueError("phonopy output supports only order 2")
        return write_phonopy(
            file, model.get(order, mapping), mapping, format=format, threshold=threshold
        )

    path = Path(file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if format == "phono3py_hdf5":
        if order != 3:
            raise ValueError("phono3py output supports only order 3")
        validate(model, mapping, order)
        write_phono3py(path, model, mapping, threshold=threshold)
    elif format == "shengbte":
        if order not in {3, 4}:
            raise ValueError("ShengBTE output supports only orders 3 and 4")
        validate(model, mapping, order)
        write_shengbte(path, model, mapping, order=order, threshold=threshold)
    elif format == "tdep":
        if order not in {2, 3, 4}:
            raise ValueError("TDEP output supports only orders 2, 3 and 4")
        if order not in model.coefficients:
            raise ValueError(f"force constants do not contain order {order}")
        if mapping is not None:
            validate(model, mapping, order)
        write_tdep(path, model, order=order, threshold=threshold)
    else:
        raise ValueError(
            f"unsupported force-constant format {format!r}; supported formats are "
            "phonopy_text, phonopy_hdf5, phono3py_hdf5, shengbte, and tdep"
        )
    logger.info(
        "Exported FC%d: format=%s threshold=%.6g file=%s elapsed=%.2f s",
        order,
        format,
        threshold,
        path,
        perf_counter() - started,
    )
    return path


__all__ = ["write"]
