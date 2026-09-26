"""Stream ASE structures into a force-fitting normal system."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import numpy as np
from ase import Atoms
from scipy.linalg.blas import dsyrk

from mlfcs.core.geometry import PeriodicGeometry
from mlfcs.core.log_error import get_logger
from mlfcs.fitting.design import ForceDesign
from mlfcs.supercell import ClusterMap

logger = get_logger(__name__)


def _sample(
    mapping: ClusterMap,
    atoms: Atoms,
    index: int,
    supercell_positions: np.ndarray,
    geometry: PeriodicGeometry,
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
    difference = atoms.positions - supercell_positions
    if not np.all(np.isfinite(difference)):
        raise ValueError(f"training structure {index} contains invalid positions")
    displacement, _ = geometry.minimum_image(difference)
    if atoms.calc is None:
        raise ValueError(f"training structure {index} has no stored ASE forces")
    forces = atoms.calc.get_property("forces", atoms, allow_calculation=False)
    if forces is None:
        raise ValueError(f"training structure {index} has no stored ASE forces")
    forces = np.asarray(forces, dtype=np.float64)
    if forces.shape != (len(atoms), 3) or not np.all(np.isfinite(forces)):
        raise ValueError(f"training structure {index} contains invalid forces")
    return displacement, forces


def _design_samples(mapping, structures, design):
    if isinstance(structures, (Atoms, np.ndarray)):
        raise TypeError("from_atoms() requires an iterable of ASE Atoms")
    supercell_positions = mapping.supercell.scaled_positions @ mapping.supercell.cell
    geometry = PeriodicGeometry(mapping.supercell.cell)
    for index, atoms in enumerate(structures):
        displacement, forces = _sample(mapping, atoms, index, supercell_positions, geometry)
        yield design.matrix(displacement), forces.reshape(-1)


def build_system(mapping: ClusterMap, structures: Iterable[Atoms]):
    """Build a :class:`FitSystem` without retaining the training structures."""
    from mlfcs.fitting.system import FitSystem

    design = ForceDesign(mapping)
    parameters = design.n_parameters
    matrix = np.zeros((parameters, parameters), dtype=np.float64, order="F")
    rhs = np.zeros(parameters, dtype=np.float64)
    force_norm = 0.0
    count = 0
    started = perf_counter()
    logger.info(
        "Fit-system construction started: mapping=%s supercell_atoms=%d orders=%s "
        "parameters=%d equations_per_structure=%d gram_memory=%.2f MiB",
        mapping.fingerprint,
        len(mapping.supercell.numbers),
        mapping.space.orders,
        parameters,
        design.rows,
        matrix.nbytes / (1024.0**2),
    )
    for count, (values, flattened) in enumerate(
        _design_samples(mapping, structures, design), start=1
    ):
        dsyrk(
            1.0,
            a=values,
            c=matrix,
            beta=1.0,
            trans=1,
            lower=0,
            overwrite_c=1,
        )
        rhs += values.T @ flattened
        force_norm += float(flattened @ flattened)
        if count == 1 or count % 10 == 0:
            elapsed = perf_counter() - started
            logger.info(
                "Fit-system accumulation: %d structures, %d cumulative force equations, "
                "%.2f s elapsed, %.2f structures/s",
                count,
                count * design.rows,
                elapsed,
                count / elapsed if elapsed else float("inf"),
            )
    if count == 0:
        raise ValueError("at least one training structure is required")
    upper = np.triu(np.asarray(matrix))
    matrix = upper + np.triu(upper, 1).T
    result = FitSystem(
        space=mapping.space,
        matrix=matrix,
        rhs=rhs,
        force_norm=force_norm,
        n_equations=count * design.rows,
        n_structures=count,
    )
    logger.info(
        "Fit system ready: %d parameters, %d structures, %d equations, force norm %.10e, "
        "fingerprint %s, %.2f s",
        parameters,
        count,
        count * design.rows,
        force_norm,
        result.fingerprint,
        perf_counter() - started,
    )
    return result


__all__ = ["build_system"]
