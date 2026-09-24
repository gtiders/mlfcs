"""Stream ASE structures into a force-fitting normal system."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import numpy as np
from ase import Atoms
from scipy.linalg.blas import dsyrk

from mlfcs.core.log import get_logger
from mlfcs.fitting.design import ForceDesign
from mlfcs.supercell import ClusterMap

logger = get_logger(__name__)


def _sample(
    mapping: ClusterMap,
    atoms: Atoms,
    index: int,
    supercell_positions: np.ndarray,
    inverse_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(atoms, Atoms):
        raise TypeError(f"training structure {index} is not an ASE Atoms object")
    supercell = mapping.supercell
    if not np.array_equal(atoms.numbers, supercell.numbers):
        raise ValueError(f"training structure {index} has a different atom sequence")
    if not np.array_equal(atoms.pbc, np.ones(3, dtype=bool)):
        raise ValueError(f"training structure {index} must be periodic in all directions")
    cell = np.asarray(atoms.cell, dtype=np.float64)
    cell_residual = float(np.max(np.linalg.norm(cell - supercell.cell, axis=1)))
    symprec = mapping.space.primitive.symprec
    if cell_residual >= symprec:
        raise ValueError(
            f"training structure {index} has cell residual {cell_residual:.10g} Å, "
            f"not below symprec {symprec:.10g} Å"
        )
    scaled = (atoms.positions - supercell_positions) @ inverse_cell
    scaled -= np.rint(scaled)
    displacement = scaled @ supercell.cell
    if not np.all(np.isfinite(displacement)):
        raise ValueError(f"training structure {index} contains invalid positions")
    if atoms.calc is None:
        raise ValueError(f"training structure {index} has no stored ASE forces")
    forces = atoms.calc.get_property("forces", atoms, allow_calculation=False)
    if forces is None:
        raise ValueError(f"training structure {index} has no stored ASE forces")
    forces = np.asarray(forces, dtype=np.float64)
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError(f"training structure {index} contains invalid forces")
    return displacement, forces


def build_system(mapping: ClusterMap, structures: Iterable[Atoms]):
    """Build a :class:`FitSystem` without retaining the training structures."""
    from mlfcs.fitting.system import FitSystem

    if isinstance(structures, (Atoms, np.ndarray)):
        raise TypeError("FitSystem.from_atoms() requires an iterable of ASE Atoms")
    design = ForceDesign(mapping)
    parameters = design.n_parameters
    matrix = np.zeros((parameters, parameters), dtype=np.float64, order="F")
    rhs = np.zeros(parameters, dtype=np.float64)
    force_norm = 0.0
    count = 0
    started = perf_counter()
    supercell_positions = mapping.supercell.scaled_positions @ mapping.supercell.cell
    inverse_cell = np.linalg.inv(mapping.supercell.cell)
    for count, atoms in enumerate(structures, start=1):
        displacement, forces = _sample(mapping, atoms, count - 1, supercell_positions, inverse_cell)
        values = design.matrix(displacement)
        dsyrk(
            1.0,
            a=values,
            c=matrix,
            beta=1.0,
            trans=1,
            lower=0,
            overwrite_c=1,
        )
        flattened = forces.reshape(-1)
        rhs += values.T @ flattened
        force_norm += float(flattened @ flattened)
    if count == 0:
        raise ValueError("at least one training structure is required")
    upper = np.triu(np.asarray(matrix))
    matrix = upper + np.triu(upper, 1).T
    logger.info(
        "Built %d-parameter fit system from %d structures in %.2f s",
        parameters,
        count,
        perf_counter() - started,
    )
    return FitSystem(
        space=mapping.space,
        matrix=matrix,
        rhs=rhs,
        force_norm=force_norm,
        n_equations=count * design.rows,
        n_structures=count,
    )


__all__ = ["build_system"]
