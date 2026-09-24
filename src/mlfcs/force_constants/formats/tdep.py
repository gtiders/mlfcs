"""TDEP ``infile.forceconstant*`` text writers for primitive-lattice FC2–FC4."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from mlfcs.force_constants.export import clean
from mlfcs.force_constants.lattice import expand
from mlfcs.force_constants.model import ForceConstants


def write_tdep(path: Path, model: ForceConstants, *, order: int, threshold: float) -> None:
    """Write TDEP's per-primitive-atom list of Cartesian energy derivatives.

    Indices are one-based; lattice vectors are integer coefficients of the
    primitive cell. TDEP's FC2 reader additionally requires a polar flag, so
    this writer marks FC2 as nonpolar (no Born-charge correction is exported).
    """
    primitive = model.space.primitive
    expanded = expand(model, order)
    grouped: dict[
        int, list[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...], np.ndarray]]
    ] = defaultdict(list)
    seen: set[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]] = set()
    for sites, translations, tensor in zip(
        expanded.sites, expanded.translations, expanded.tensors, strict=True
    ):
        key = (sites, translations)
        if key in seen:
            raise RuntimeError("cluster-space expansion produced a duplicate lattice cluster")
        seen.add(key)
        grouped[sites[0]].append((sites, translations, clean(tensor, threshold)))

    with path.open("w", encoding="ascii") as handle:
        handle.write(f"{primitive.size}\n{model.space.block(order).cutoff:.17g}\n")
        for first in range(primitive.size):
            blocks = sorted(grouped[first], key=lambda block: (block[0], block[1]))
            handle.write(f"{len(blocks)}\n")
            for sites, translations, tensor in blocks:
                if order == 2:
                    handle.write(f"{sites[1] + 1}\n")
                    vectors = translations
                else:
                    for site in sites:
                        handle.write(f"{site + 1}\n")
                    vectors = ((0, 0, 0), *translations)
                for vector in vectors:
                    handle.write(" ".join(str(int(value)) for value in vector) + "\n")
                for directions in product(range(3), repeat=order - 1):
                    handle.write(" ".join(f"{value:.17e}" for value in tensor[directions]) + "\n")
        if order == 2:
            handle.write("0\n")


__all__ = ["write_tdep"]
