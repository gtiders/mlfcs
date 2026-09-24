"""Focused ASE contracts for order-selected Taylor evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from mlfcs import ClusterMap, ForceConstants, PrimitiveCell, Supercell, build_cluster_space
from mlfcs.fitting.design import ForceDesign
from mlfcs.tools import TaylorCalculator, build_supercell


@pytest.fixture(scope="module")
def model_and_map() -> tuple[ForceConstants, ClusterMap]:
    atoms = bulk("Ar", "sc", a=3.0)
    primitive = PrimitiveCell.from_atoms(atoms, symprec=1e-5)
    matrix = np.diag([3, 3, 3])
    supercell = Supercell.from_atoms(
        primitive,
        build_supercell(atoms, matrix),
        matrix=matrix,
    )
    space = build_cluster_space(
        primitive,
        cutoffs={2: 3.1, 3: 3.1, 4: 3.1},
        max_body_orders={2: 2, 3: 3, 4: 4},
    )
    coefficients = {
        order: np.random.default_rng(order).normal(
            size=space.block(order).parameters.stop - space.block(order).parameters.start
        )
        for order in space.orders
    }
    return ForceConstants(space, coefficients), ClusterMap.build(space, supercell)


def _evaluate(
    model: ForceConstants,
    mapping: ClusterMap,
    displacement: np.ndarray,
    orders: tuple[int, ...] | None = None,
) -> tuple[float, np.ndarray]:
    calculator = TaylorCalculator(model, mapping, orders=orders)
    atoms = calculator.supercell
    atoms.positions += displacement
    atoms.calc = calculator
    return atoms.get_potential_energy(), atoms.get_forces()


def test_taylor_calculator_masks_orders_and_matches_the_force_design(model_and_map) -> None:
    model, mapping = model_and_map
    count = len(mapping.supercell.numbers)
    displacement = np.random.default_rng(7).normal(scale=0.02, size=(count, 3))
    full_energy, full_forces = _evaluate(model, mapping, displacement)
    pieces = {order: _evaluate(model, mapping, displacement, (order,)) for order in (2, 3, 4)}
    masked_energy, masked_forces = _evaluate(model, mapping, displacement, (2, 4))
    quartic_energy, quartic_forces = pieces[4]

    np.testing.assert_allclose(
        full_forces, sum(forces for _, forces in pieces.values()), atol=1e-12
    )
    assert full_energy == pytest.approx(sum(energy for energy, _ in pieces.values()))
    np.testing.assert_allclose(masked_forces, pieces[2][1] + quartic_forces, atol=1e-12)
    assert masked_energy == pytest.approx(pieces[2][0] + quartic_energy)
    assert quartic_energy != 0.0

    design = ForceDesign(mapping).matrix(displacement)
    np.testing.assert_allclose(full_forces.ravel(), design @ model.parameters(), atol=1e-12)


def test_taylor_energy_derivative_is_the_negative_force(model_and_map) -> None:
    model, mapping = model_and_map
    displacement = np.random.default_rng(9).normal(
        scale=0.02, size=(len(mapping.supercell.numbers), 3)
    )
    for orders in ((2, 3, 4), (4,)):
        _, forces = _evaluate(model, mapping, displacement, orders)
        for atom, axis in ((0, 0), (3, 2)):
            step = np.zeros_like(displacement)
            step[atom, axis] = 1e-5
            higher = _evaluate(model, mapping, displacement + step, orders)[0]
            lower = _evaluate(model, mapping, displacement - step, orders)[0]
            assert -(higher - lower) / (2e-5) == pytest.approx(
                forces[atom, axis], rel=1e-8, abs=1e-10
            )


def test_numba_kernels_are_specialized_to_each_force_constant_model(model_and_map) -> None:
    model, mapping = model_and_map
    doubled = ForceConstants(model.space, {2: 2.0 * model.coefficients[2]})
    first = TaylorCalculator(model, mapping, orders=(2,))
    second = TaylorCalculator(doubled, mapping)
    atoms = first.supercell
    atoms.positions[0, 0] += 0.03
    atoms.calc = first
    force = atoms.get_forces()
    atoms.calc = second
    np.testing.assert_allclose(atoms.get_forces(), 2.0 * force, atol=1e-12)
    assert first._kernels[0] is not second._kernels[0]
    assert first._kernels[0].nopython_signatures
    assert second._kernels[0].nopython_signatures


def test_taylor_calculator_uses_a_fixed_supercell_and_immutable_order_choice(model_and_map) -> None:
    model, mapping = model_and_map
    calculator = TaylorCalculator(model, mapping, orders=(4,))
    supercell = calculator.supercell
    assert not hasattr(calculator, "reference")
    supercell.calc = calculator
    assert supercell.get_potential_energy() == 0.0
    np.testing.assert_array_equal(supercell.get_forces(), 0.0)
    assert calculator._kernels[0].nopython_signatures
    assert calculator.orders == (4,)
    with pytest.raises(AttributeError):
        calculator.orders = (2, 3, 4)
    with pytest.raises(ValueError, match="unique and ascending"):
        TaylorCalculator(model, mapping, orders=(4, 2))
    with pytest.raises(ValueError, match="do not contain"):
        TaylorCalculator(model, mapping, orders=(5,))
    strained = calculator.supercell
    strained.cell[0, 0] += 0.01
    strained.calc = calculator
    with pytest.raises(ValueError, match="fixed supercell"):
        strained.get_forces()
