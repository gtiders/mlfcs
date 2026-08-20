from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from supercell_helpers import make_supercell

from mlfcs.anharmonic.sscha import SSCHA


class TranslationalHarmonic(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, reference: Atoms, spring: float = 2.0):
        super().__init__()
        self.reference = reference.copy()
        self.spring = spring

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        delta = atoms.positions - self.reference.positions
        delta -= np.rint(delta @ np.linalg.inv(atoms.cell)) @ atoms.cell
        delta -= np.mean(delta, axis=0)
        self.results = {
            "energy": self.spring * np.sum(delta**2) / 2,
            "forces": -self.spring * delta,
        }


def primitive() -> Atoms:
    return Atoms("Al", cell=np.eye(3) * 4.0, scaled_positions=[[0, 0, 0]], pbc=True)


def test_sscha_has_no_external_sow_reap_api():
    assert not hasattr(SSCHA, "sow")
    assert not hasattr(SSCHA, "reap")


def test_sscha_direct_run_and_linear_mixing(tmp_path):
    n_atoms = 8
    spring = 2.0
    fc = np.zeros((n_atoms, n_atoms, 3, 3))
    for axis in range(3):
        fc[:, :, axis, axis] = spring * (np.eye(n_atoms) - np.ones((n_atoms, n_atoms)) / n_atoms)
    primitive_atoms = primitive()
    direct = SSCHA(
        primitive_atoms,
        reference=make_supercell(primitive_atoms, (2, 2, 2))[0],
        snapshots=16,
        max_iterations=0,
        random_seed=11,
        initial_force_constants=fc,
    )
    calc = TranslationalHarmonic(direct.supercell_atoms, spring=spring)

    direct.step(calc, calculate_free_energy=False)

    assert len(direct.history) == 1
    assert all(item.sampling == "canonical" for item in direct.history)
    assert all(item.fitting_relative_force_error < 1e-7 for item in direct.history)
    assert all(item.relative_force_constant_change is not None for item in direct.history)
    assert all(
        item.relative_force_constant_change == pytest.approx(item.raw_relative_force_constant_change)
        for item in direct.history
    )
    target = tmp_path / "fc2.hdf5"
    direct.write(target)
    assert target.is_file()

    raw = SSCHA(
        primitive_atoms,
        reference=make_supercell(primitive_atoms, (2, 2, 2))[0],
        snapshots=16,
        max_iterations=0,
        random_seed=11,
        initial_force_constants=fc,
    )
    raw.step(calc, calculate_free_energy=False)
    mixed = SSCHA(
        primitive_atoms,
        reference=make_supercell(primitive_atoms, (2, 2, 2))[0],
        snapshots=16,
        max_iterations=0,
        random_seed=11,
        initial_force_constants=fc,
        mixing=0.25,
    )
    mixed.step(calc, calculate_free_energy=False)
    np.testing.assert_allclose(
        mixed.force_constants,
        0.75 * fc + 0.25 * raw.force_constants,
    )
    assert mixed.history[0].relative_force_constant_change == pytest.approx(
        0.25 * mixed.history[0].raw_relative_force_constant_change
    )


def test_sscha_temperature_schedule_sorts_and_returns_independent_results():
    primitive_atoms = primitive()
    reference = make_supercell(primitive_atoms, (2, 2, 2))[0]
    initial = np.zeros((len(reference), len(reference), 3, 3))
    for axis in range(3):
        initial[:, :, axis, axis] = 2.0 * (
            np.eye(len(reference)) - np.ones((len(reference), len(reference))) / len(reference)
        )
    result = SSCHA(
        primitive_atoms,
        reference=reference,
        temperature=[600, 300],
        snapshots=2,
        max_iterations=0,
        initial_force_constants=initial,
        random_seed=23,
    ).run(TranslationalHarmonic(reference), calculate_free_energy=False)

    assert result.temperatures == (300.0, 600.0)
    assert result.continuation
    assert result.at_temperature(300).temperature == 300.0
    assert len(result.at_temperature(600).history) == 1


@pytest.mark.parametrize("mixing", [0.0, -0.1, 1.1])
def test_sscha_validates_linear_mixing(mixing):
    primitive_atoms = primitive()
    reference = make_supercell(primitive_atoms, (2, 2, 2))[0]
    with pytest.raises(ValueError, match="mixing"):
        SSCHA(primitive_atoms, reference=reference, snapshots=1, mixing=mixing)


def test_sscha_accepts_a_reordered_nondiagonal_reference_frame():
    cell = primitive()
    reference, _ = make_supercell(cell, [[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    reference = reference[[1, 0]]
    sscha = SSCHA(cell, reference=reference, snapshots=2, max_iterations=0)

    np.testing.assert_array_equal(sscha.supercell_atoms.numbers, reference.numbers)
    assert sscha._index.representative(0) == 1
