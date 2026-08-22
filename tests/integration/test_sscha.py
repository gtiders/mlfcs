from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mlfcs.sscha import SSCHA

pytestmark = pytest.mark.integration


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


def test_sscha_external_sow_reap():
    sscha = SSCHA(
        primitive(),
        supercell=(2, 2, 2),
        snapshots=16,
        max_iterations=0,
        random_seed=7,
    )
    structures = sscha.sow()
    assert [a.info["mlfcs_configuration_id"] for a in structures] == list(range(16))
    calc = TranslationalHarmonic(sscha.supercell_atoms)
    forces = {}
    energies = {}
    for i, atoms in enumerate(structures):
        atoms.calc = calc
        forces[i] = atoms.get_forces()
        energies[i] = atoms.get_potential_energy()

    result = sscha.reap(forces, energies=energies, reference_energy=0.0)

    assert result.index == 0
    assert result.sampling == "cartesian"
    assert result.force_constants.shape == (8, 8, 3, 3)
    assert np.isfinite(result.force_constants).all()
    assert result.free_energy is None
    assert result.free_energy_error is None
    assert result.potential_energy is not None and np.isfinite(result.potential_energy)
    assert result.fitting_relative_force_error < 1e-12
    assert result.relative_force_constant_change is None


def test_sscha_direct_run_and_average(tmp_path):
    n_atoms = 8
    spring = 2.0
    fc = np.zeros((n_atoms, n_atoms, 3, 3))
    for axis in range(3):
        fc[:, :, axis, axis] = spring * (np.eye(n_atoms) - np.ones((n_atoms, n_atoms)) / n_atoms)
    sscha = SSCHA(
        primitive(),
        supercell=(2, 2, 2),
        snapshots=16,
        max_iterations=1,
        random_seed=11,
        initial_force_constants=fc,
    )
    calc = TranslationalHarmonic(sscha.supercell_atoms, spring=spring)

    returned = sscha.run(calc, calculate_free_energy=False)

    assert returned is sscha
    assert len(sscha.history) == 2
    assert all(item.sampling == "canonical" for item in sscha.history)
    assert all(item.fitting_relative_force_error < 1e-7 for item in sscha.history)
    assert all(item.relative_force_constant_change is not None for item in sscha.history)
    averaged = sscha.use_average(2)
    assert averaged.shape == fc.shape
    target = tmp_path / "fc2.hdf5"
    sscha.write(target)
    assert target.is_file()


def test_sscha_validates_external_order():
    sscha = SSCHA(primitive(), supercell=(2, 2, 2), snapshots=2, max_iterations=0)
    sscha.sow()
    with pytest.raises(ValueError, match="IDs"):
        sscha.reap({0: np.zeros((8, 3))})


def test_sscha_canonical_iterations_use_independent_reproducible_seeds():
    n_atoms = 8
    fc = np.zeros((n_atoms, n_atoms, 3, 3))
    for axis in range(3):
        fc[:, :, axis, axis] = 2.0 * (np.eye(n_atoms) - np.ones((n_atoms, n_atoms)) / n_atoms)

    def snapshots(iteration: int):
        sscha = SSCHA(
            primitive(),
            supercell=(2, 2, 2),
            snapshots=4,
            max_iterations=2,
            random_seed=42,
            initial_force_constants=fc,
        )
        sscha.history.extend([None] * iteration)
        return np.asarray([atoms.positions for atoms in sscha.sow()])

    first = snapshots(0)
    second = snapshots(1)
    assert not np.allclose(first, second)
    np.testing.assert_allclose(first, snapshots(0), atol=0, rtol=0)
