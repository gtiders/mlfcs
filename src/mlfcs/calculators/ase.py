"""ASE adapter for canonical Taylor force constants."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mlfcs.calculators.taylor import TaylorPotential
from mlfcs.force_constants.representation import ForceConstants
from mlfcs.io.hdf5 import read_hdf5

logger = logging.getLogger(__name__)


class MLFCSCalculator(Calculator):
    """Evaluate a fixed-cell, reference-relative Taylor IFC potential.

    Energies are measured from the supplied reference structure with
    :math:`E_0=0`; the linear force-constant term is also defined as zero.
    Only energy and forces are implemented.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(
        self,
        force_constants: ForceConstants,
        *,
        reference: Atoms | None = None,
        maximum_displacement: float | None = None,
    ) -> None:
        super().__init__()
        self.potential = TaylorPotential(
            force_constants,
            reference=reference,
            maximum_displacement=maximum_displacement,
        )
        self._warned_displacement = False

    @classmethod
    def from_hdf5(
        cls,
        source: str | Path,
        *,
        reference: Atoms | None = None,
        maximum_displacement: float | None = None,
    ) -> MLFCSCalculator:
        """Create a calculator from native MLFCS HDF5 force constants."""
        return cls(
            read_hdf5(source),
            reference=reference,
            maximum_displacement=maximum_displacement,
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: tuple[str, ...] | list[str] = ("energy", "forces"),
        system_changes: list[str] = all_changes,
    ) -> None:
        """Evaluate requested ASE properties for one compatible structure."""
        unsupported = set(properties) - set(self.implemented_properties)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(f"MLFCSCalculator does not implement: {names}")
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("atoms are required for calculation")
        displacement = self.potential.displacement(atoms)
        limit = self.potential.maximum_displacement
        maximum = float(np.max(np.linalg.norm(displacement, axis=1), initial=0.0))
        if limit is not None and maximum > limit and not self._warned_displacement:
            logger.warning(
                "Maximum displacement %.6f A exceeds the Taylor validity warning threshold "
                "%.6f A; evaluation continues without clipping",
                maximum,
                limit,
            )
            self._warned_displacement = True
        energy, forces = self.potential.evaluate_displacement(displacement)
        self.results = {"energy": energy, "forces": forces}

    def force_design_batch(self, displacements: np.ndarray) -> np.ndarray:
        """Evaluate forces for a batch of reference-relative displacements."""
        values = np.asarray(displacements, dtype=float)
        if values.ndim != 3:
            raise ValueError("displacements must have shape (batch, atoms, 3)")
        return np.asarray(
            [self.potential.evaluate_displacement(value)[1] for value in values]
        )


__all__ = ["MLFCSCalculator"]
