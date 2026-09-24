"""ShengBTE-style anharmonic text writer."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from mlfcs.force_constants.export import clean, validate
from mlfcs.force_constants.lattice import expand
from mlfcs.force_constants.model import ForceConstants
from mlfcs.supercell import ClusterMap


def _vector_line(vector: np.ndarray) -> str:
    return " ".join(f"{value:>15.10e}" for value in vector)


def write_shengbte(
    path: Path,
    model: ForceConstants,
    mapping: ClusterMap,
    *,
    order: int,
    threshold: float,
) -> None:
    """Write nonzero exact-lattice FC3 or FC4 blocks in ShengBTE style."""
    validate(model, mapping, order)
    expanded = expand(model, order)
    physical: dict[
        tuple[int, ...], tuple[tuple[int, ...], tuple[tuple[int, int, int], ...], np.ndarray]
    ] = {}
    for sites, translations, tensor in zip(
        expanded.sites, expanded.translations, expanded.tensors, strict=True
    ):
        key = (*sites, *(value for translation in translations for value in translation))
        if key in physical:
            raise RuntimeError("cluster-space expansion produced a duplicate lattice cluster")
        physical[key] = (sites, translations, np.array(tensor, copy=True))

    blocks = []
    primitive_cell = model.space.primitive.cell
    for key in sorted(physical):
        sites, translations, tensor = physical[key]
        tensor = clean(tensor, threshold)
        if not np.any(tensor):
            continue
        lines = ["", f"{len(blocks) + 1:>5}"]
        lines.extend(
            _vector_line(np.asarray(translation) @ primitive_cell) for translation in translations
        )
        lines.append(" ".join(f"{site + 1:>6d}" for site in sites))
        for directions in product(range(3), repeat=order):
            direction_text = " ".join(f"{direction + 1:>2d}" for direction in directions)
            lines.append(f"{direction_text} {tensor[directions]:>20.10e}")
        blocks.append("\n".join(lines) + "\n")
    path.write_text(f"{len(blocks):>5}\n" + "".join(blocks))


__all__ = ["write_shengbte"]
