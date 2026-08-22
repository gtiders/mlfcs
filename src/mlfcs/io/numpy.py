from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mlfcs.model import ForceConstants


def write_numpy(target: str | Path, force_constants: ForceConstants) -> None:
    arrays = {f"fc{order}": force_constants.materialize(order) for order in force_constants.orders}
    arrays.update(
        cell=np.asarray(force_constants.supercell.cell),
        positions=force_constants.supercell.positions,
        numbers=force_constants.supercell.numbers,
    )
    np.savez_compressed(target, **arrays)
