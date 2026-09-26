"""Reconstruct primitive force constants from evaluated ASE structures."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from mlfcs.core.geometry import PeriodicGeometry
from mlfcs.core.log_error import get_logger
from mlfcs.force_constants import ForceConstants

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mlfcs.finite_difference.difference import FiniteDifference


def _forces(difference: FiniteDifference, structures: Sequence[Atoms]) -> np.ndarray:
    if isinstance(structures, (Atoms, np.ndarray)) or not isinstance(structures, Sequence):
        raise TypeError("reconstruct() accepts only an ordered sequence of ASE Atoms")
    if len(structures) != difference.n_configurations:
        raise ValueError(
            f"expected {difference.n_configurations} structures, got {len(structures)}; "
            "structure i must correspond to displacements()[i]"
        )
    expected = difference.displacements()
    cell = difference.mapping.supercell.cell
    geometry = PeriodicGeometry(cell)
    symprec = difference.mapping.space.primitive.symprec
    forces = []
    for index, (atoms, target) in enumerate(zip(structures, expected, strict=True)):
        if not isinstance(atoms, Atoms):
            raise TypeError(f"structure {index} is not an ASE Atoms object")
        if not np.array_equal(atoms.numbers, target.numbers):
            raise ValueError(f"structure {index} has a different atom sequence")
        if not np.array_equal(atoms.pbc, target.pbc):
            raise ValueError(f"structure {index} has different periodic boundary conditions")
        cell_residual = float(
            np.max(np.linalg.norm(np.asarray(atoms.cell) - np.asarray(target.cell), axis=1))
        )
        _, distances = geometry.minimum_image(atoms.positions - target.positions)
        position_residual = float(np.max(distances))
        if cell_residual >= symprec or position_residual >= symprec:
            raise ValueError(
                f"structure {index} does not match displacements()[{index}]: cell residual "
                f"{cell_residual:.10g} Å, position residual {position_residual:.10g} Å, "
                f"symprec {symprec:.10g} Å"
            )
        if atoms.calc is None:
            raise ValueError(f"structure {index} has no stored ASE forces")
        values = atoms.calc.get_property("forces", atoms, allow_calculation=False)
        if values is None:
            raise ValueError(f"structure {index} has no stored ASE forces")
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(atoms), 3) or not np.all(np.isfinite(array)):
            raise ValueError(f"structure {index} contains invalid forces")
        forces.append(array)
    return np.asarray(forces)


def _weights(disps: tuple[float, ...]) -> np.ndarray:
    """Return the unique even-error extrapolation weights at zero displacement."""
    squared = np.square(np.asarray(disps, dtype=np.float64))
    weights = np.ones(len(squared), dtype=np.float64)
    for index, value in enumerate(squared):
        for other, other_value in enumerate(squared):
            if other != index:
                weights[index] *= -other_value / (value - other_value)
    return weights


def reconstruct(difference: FiniteDifference, structures: Sequence[Atoms]) -> ForceConstants:
    started = perf_counter()
    logger.info(
        "FC%d reconstruction started: %d configurations, %d displacements, %d observed keys",
        difference.order,
        len(structures),
        len(difference.disps),
        len(difference._keys),
    )
    values = _forces(difference, structures)
    order = difference.order
    signs = np.asarray(difference._signs, dtype=np.float64)
    sign_weights = np.prod(signs, axis=1)
    disp_weights = _weights(difference.disps)
    sign_count = len(signs)
    disp_count = len(difference.disps)
    derivatives: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for key_index, key in enumerate(difference._keys):
        estimates = []
        base = key_index * disp_count * sign_count
        for disp_index, disp in enumerate(difference.disps):
            begin = base + disp_index * sign_count
            estimates.append(
                -np.tensordot(
                    sign_weights,
                    values[begin : begin + sign_count],
                    axes=(0, 0),
                )
                / (2.0 * disp) ** (order - 1)
            )
        derivatives[key] = np.tensordot(disp_weights, estimates, axes=(0, 0))

    mapping = difference.mapping
    space = mapping.space
    block = space.block(order)
    coefficients = []
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        orbit = space.orbits[orbit_index]
        image = orbit.clusters.index(orbit.representative)
        atoms = tuple(int(value) for value in mapping.atoms[orbit_index][image])
        observed = []
        for row in orbit.observation_rows:
            directions = np.unravel_index(int(row), (3,) * order)
            key = tuple((atoms[axis], int(directions[axis])) for axis in range(order - 1))
            observed.append(derivatives[key][atoms[-1], int(directions[-1])])
        coefficients.append(
            np.linalg.solve(orbit.observation_matrix, np.asarray(observed, dtype=np.float64))
        )
    model = ForceConstants(space, {order: np.concatenate(coefficients)})
    logger.info(
        "FC%d reconstruction complete: %d parameters, force_rms=%.10e, "
        "force_constants_fingerprint=%s, %.2f s",
        order,
        len(model.coefficients[order]),
        float(np.sqrt(np.mean(values**2))),
        model.fingerprint,
        perf_counter() - started,
    )
    return model


__all__ = ["reconstruct"]
