"""Order-defined finite differences over ASE structures."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs.core.log import get_logger
from mlfcs.finite_difference.displacements import Displacements
from mlfcs.finite_difference.reconstruction import reconstruct
from mlfcs.force_constants import ForceConstants
from mlfcs.supercell import ClusterMap

logger = get_logger(__name__)


def _keys(mapping: ClusterMap, order: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    block = mapping.space.block(order)
    keys: set[tuple[tuple[int, int], ...]] = set()
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        orbit = mapping.space.orbits[orbit_index]
        image = orbit.clusters.index(orbit.representative)
        atoms = tuple(int(value) for value in mapping.atoms[orbit_index][image])
        for row in orbit.observation_rows:
            directions = np.unravel_index(int(row), (3,) * order)
            keys.add(tuple((atoms[axis], int(directions[axis])) for axis in range(order - 1)))
    return tuple(sorted(keys))


class FiniteDifference:
    """One finite-difference experiment whose ASE structures define its data."""

    __slots__ = ("_keys", "_signs", "disps", "mapping", "order")

    def __init__(
        self,
        mapping: ClusterMap,
        *,
        order: int,
        disps: float | Sequence[float] = 0.01,
    ):
        order = int(order)
        if order < 2:
            raise ValueError("force-constant order must be at least 2")
        source = (disps,) if np.isscalar(disps) else disps
        values = tuple(sorted(float(disp) for disp in source))
        if not values:
            raise ValueError("finite differences require at least one step")
        if any(not np.isfinite(disp) or disp <= 0.0 for disp in values):
            raise ValueError("finite-difference displacements must be positive finite lengths")
        if len(set(values)) != len(values):
            raise ValueError("finite-difference displacements must be distinct")
        mapping.space.block(order)
        mapping.rank_info(order).require_full()
        self.mapping = mapping
        self.order = order
        self.disps = values
        self._keys = _keys(mapping, order)
        self._signs = np.asarray(list(product((-1, 1), repeat=order - 1)), dtype=np.int8)
        self._signs.setflags(write=False)
        logger.info(
            "Prepared FC%d finite difference: %d configurations at displacements %s Å",
            order,
            self.n_configurations,
            ", ".join(f"{disp:.10g}" for disp in self.disps),
        )

    @property
    def n_configurations(self) -> int:
        return len(self._keys) * len(self.disps) * len(self._signs)

    def displacements(self) -> Sequence[Atoms]:
        """Return the canonical key-displacement-sign sequence of structures."""
        return Displacements(self)

    def evaluate(self, calculator: Any) -> tuple[Atoms, ...]:
        """Force a fresh calculation for every displacement and freeze its forces."""
        evaluated = []
        for atoms in self.displacements():
            atoms.calc = calculator
            calculator.calculate(
                atoms=atoms,
                properties=["forces"],
                system_changes=all_changes,
            )
            forces = calculator.get_property("forces", atoms, allow_calculation=False)
            if forces is None:
                raise ValueError("calculator did not produce forces")
            values = np.asarray(forces, dtype=np.float64)
            if values.shape != (len(atoms), 3) or not np.all(np.isfinite(values)):
                raise ValueError("calculator returned invalid forces")
            atoms.calc = SinglePointCalculator(atoms, forces=values)
            evaluated.append(atoms)
        return tuple(evaluated)

    def reconstruct(self, structures: Sequence[Atoms]) -> ForceConstants:
        """Reconstruct from the canonical sequence of evaluated ASE structures."""
        return reconstruct(self, structures)


__all__ = ["FiniteDifference"]
