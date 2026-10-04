"""Finite-difference experiments, displacement sequences and reconstruction."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import product
from time import perf_counter
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs.core.log import get_logger
from mlfcs.force_constants import ForceConstants
from mlfcs.mapping import ClusterMap

logger = get_logger(__name__)


def _keys(cluster_map: ClusterMap, order: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return sorted unique displacement-coordinate tuples needed by representative observation
    rows.
    """
    block = cluster_map.cluster_space.block(order)
    keys: set[tuple[tuple[int, int], ...]] = set()
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        orbit = cluster_map.cluster_space.orbits[orbit_index]
        image = orbit.clusters.index(orbit.representative)
        atoms = tuple(int(value) for value in cluster_map.image_atom_indices[orbit_index][image])
        for row in orbit.observation_rows:
            directions = np.unravel_index(int(row), (3,) * order)
            keys.add(tuple((atoms[axis], int(directions[axis])) for axis in range(order - 1)))
    return tuple(sorted(keys))


class FiniteDifference:
    """Central finite-difference experiment on one structurally full-rank ClusterMap.

    Parameters
    ----------
    cluster_map : ClusterMap
        Reference supercell relation, retained by reference.
    order : int
        Included force-constant tensor order, at least two.
    disps : float or sequence of float, default 0.01
        Distinct positive step lengths in angstrom. Sorted steps are combined
        by extrapolating the even central-difference error to zero step.

    Notes
    -----
    Samples are ordered by displacement key, then ascending step, then sign.
    Repeated atom/axis entries add displacements. Construction requires full
    folded rank but does not calculate forces. evaluate explicitly calculates;
    reconstruct reads only stored forces from the canonical sequence.

    Raises
    ------
    ValueError
        Steps or order are invalid.
    AliasingError
        Supercell folding loses required parameters.
    """

    __slots__ = ("_keys", "_signs", "cluster_map", "disps", "order")

    def __init__(
        self,
        cluster_map: ClusterMap,
        *,
        order: int,
        disps: float | Sequence[float] = 0.01,
    ):
        """Validate steps and folded rank, then prepare canonical displacement keys and signs."""
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
        cluster_map.cluster_space.block(order)
        cluster_map.rank_info(order).require_full()
        self.cluster_map = cluster_map
        self.order = order
        self.disps = values
        self._keys = _keys(cluster_map, order)
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
        """Number of required structures: keys times step lengths times 2**(order-1)."""
        return len(self._keys) * len(self.disps) * len(self._signs)

    def displacements(self) -> Sequence[Atoms]:
        """Return the canonical key-displacement-sign sequence of structures."""
        return Displacements(self)

    def evaluate(self, calculator: Any) -> tuple[Atoms, ...]:
        """Evaluate every canonical displacement and freeze its forces in ASE snapshots.

        calculator is an ASE-compatible calculator reused for fresh calculate calls.
        Return a tuple of independent Atoms with SinglePointCalculator forces in
        eV/angstrom. Invalid/missing force results raise ValueError; reference
        geometry is unchanged. This method intentionally performs calculations.
        """
        started = perf_counter()
        logger.info(
            "Finite-difference evaluation started: order=%d configurations=%d steps_angstrom=%s",
            self.order,
            self.n_configurations,
            self.disps,
        )
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
            count = len(evaluated)
            if count == 1 or count % 10 == 0:
                logger.info(
                    "Finite-difference evaluation progress: configurations=%d total=%d elapsed_s=%.2f",
                    count,
                    self.n_configurations,
                    perf_counter() - started,
                )
        logger.info(
            "Finite-difference evaluation complete: configurations=%d elapsed_s=%.2f",
            len(evaluated),
            perf_counter() - started,
        )
        return tuple(evaluated)

    def reconstruct(self, structures: Sequence[Atoms]) -> ForceConstants:
        """Return the selected-order ForceConstants from canonical evaluated snapshots.

        structures must be an ordered Sequence matching displacements(), with stored
        (n_atoms, 3) forces. Missing data or geometry/order mismatches raise
        ValueError; calculators are never evaluated. Central mixed derivatives and
        even-error extrapolation recover representative component parameters.
        """
        started = perf_counter()
        logger.info(
            "Finite-difference reconstruction started: order=%d configurations=%d steps_angstrom=%s",
            self.order,
            self.n_configurations,
            self.disps,
        )
        model = _reconstruct(self, structures)
        logger.info(
            "Finite-difference reconstruction complete: order=%d parameters=%d elapsed_s=%.2f",
            self.order,
            len(model.coefficients[self.order]),
            perf_counter() - started,
        )
        return model


class Displacements(Sequence[Atoms]):
    """The canonical key-displacement-sign structures generated by one experiment."""

    __slots__ = ("_difference",)

    def __init__(self, difference: FiniteDifference):
        """Reference an experiment without materializing its displaced structures."""
        self._difference = difference

    def __len__(self) -> int:
        """Return the experiment configuration count."""
        return self._difference.n_configurations

    def __getitem__(self, index: int | slice) -> Atoms | tuple[Atoms, ...]:
        """Generate a fresh displaced ASE structure, or a tuple for a slice.

        Supports negative indices. Adds mlfcs_id, mlfcs_disp and the Cartesian
        mlfcs_displacement array; generated atoms have no evaluated forces.
        Out-of-range indices raise IndexError.
        """
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("displacement index is outside the finite difference")
        difference = self._difference
        sign_count = len(difference._signs)
        disp_count = len(difference.disps)
        key_index, remainder = divmod(position, disp_count * sign_count)
        disp_index, sign_index = divmod(remainder, sign_count)
        key = difference._keys[key_index]
        disp = difference.disps[disp_index]
        delta = np.zeros((len(difference.cluster_map.atomic_numbers), 3))
        for sign, (atom, axis) in zip(difference._signs[sign_index], key, strict=True):
            delta[int(atom), int(axis)] += int(sign) * disp
        atoms = difference.cluster_map.supercell_atoms
        atoms.positions += delta
        atoms.info["mlfcs_id"] = position
        atoms.info["mlfcs_disp"] = disp
        atoms.arrays["mlfcs_displacement"] = delta
        return atoms

    def __iter__(self) -> Iterator[Atoms]:
        """Generate fresh structures in canonical key-step-sign order."""
        for index in range(len(self)):
            yield self[index]


def _forces(difference: FiniteDifference, structures: Sequence[Atoms]) -> np.ndarray:
    """Validate the canonical snapshot sequence and gather finite stored forces without calculation."""
    if isinstance(structures, (Atoms, np.ndarray)) or not isinstance(structures, Sequence):
        raise TypeError("reconstruct() accepts only an ordered sequence of ASE Atoms")
    if len(structures) != difference.n_configurations:
        raise ValueError(
            f"expected {difference.n_configurations} structures, got {len(structures)}; "
            "structure i must correspond to displacements()[i]"
        )
    expected = difference.displacements()
    cell = difference.cluster_map.cell
    inverse = np.linalg.inv(cell)
    symprec = difference.cluster_map.cluster_space.symprec
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
        difference_scaled = (atoms.positions - target.positions) @ inverse
        difference_scaled -= np.rint(difference_scaled)
        position_residual = float(np.max(np.linalg.norm(difference_scaled @ cell, axis=1)))
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


def _reconstruct(difference: FiniteDifference, structures: Sequence[Atoms]) -> ForceConstants:
    """Combine signed central-force differences, extrapolate steps and bind observed components.

    The leading minus sign converts force derivatives to energy derivatives.
    Repeated displacement directions are already encoded in each stencil key.
    Return only the experiment's order; input structures are not modified.
    """
    values = _forces(difference, structures)
    order = difference.order
    signs = np.asarray(difference._signs, dtype=np.float64)
    # The mixed central stencil weights each force by the product of its
    # displacement signs; the later minus converts force to energy derivatives.
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

    cluster_map = difference.cluster_map
    space = cluster_map.cluster_space
    block = space.block(order)
    coefficients = []
    for orbit_index in range(block.orbits.start, block.orbits.stop):
        orbit = space.orbits[orbit_index]
        image = orbit.clusters.index(orbit.representative)
        atoms = tuple(int(value) for value in cluster_map.image_atom_indices[orbit_index][image])
        observed = []
        for row in orbit.observation_rows:
            directions = np.unravel_index(int(row), (3,) * order)
            key = tuple((atoms[axis], int(directions[axis])) for axis in range(order - 1))
            observed.append(derivatives[key][atoms[-1], int(directions[-1])])
        coefficients.append(
            np.linalg.solve(orbit.observation_matrix, np.asarray(observed, dtype=np.float64))
        )
    return ForceConstants(space, {order: np.concatenate(coefficients)})


__all__ = ["FiniteDifference"]
