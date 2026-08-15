from typing import ClassVar

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from mlfcs import ForceConstantCalculation
from mlfcs.runtime import configure_jax

pytestmark = pytest.mark.integration


class ZeroCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["forces"]

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["forces"] = np.zeros((len(atoms), 3))


def calculation():
    return ForceConstantCalculation(
        bulk("Si", "diamond", a=5.43),
        order=3,
        supercell=(2, 2, 2),
        cutoff=-1,
    )


def test_sow_ids_define_positional_reap_order():
    job = calculation()
    structures = job.sow()
    assert [atoms.info["mlfcs_configuration_id"] for atoms in structures] == list(
        range(len(structures))
    )
    assert all(atoms.info["mlfcs_atom_order"] == "reference" for atoms in structures)

    forces = np.zeros((len(structures), len(job.supercell), 3))
    positional = job.reap(forces)[3]
    mapped = job.reap(
        {index: forces[index] for index in reversed(range(len(structures)))},
    )[3]
    np.testing.assert_array_equal(positional, mapped)


def test_reap_rejects_missing_ids():
    job = calculation()
    force = np.zeros((len(job.supercell), 3))
    with pytest.raises(ValueError, match="missing"):
        job.reap({0: force})


def test_reference_force_order_and_user_calculator_path():
    job = calculation()
    structures = job.sow()
    assert all(atoms.info["mlfcs_atom_order"] == "reference" for atoms in structures)
    forces = np.zeros((len(structures), len(job.supercell), 3))
    result = job.reap(forces)
    np.testing.assert_array_equal(result[3], 0.0)
    evaluated = job.evaluate(ZeroCalculator())
    assert evaluated.shape == (len(job.plan), len(job.supercell), 3)
    direct = job.reap(evaluated)
    np.testing.assert_array_equal(direct[3], 0.0)


def test_second_order_uses_the_same_pipeline():
    job = ForceConstantCalculation(
        bulk("Si", "diamond", a=5.43),
        order=2,
        supercell=(2, 2, 2),
        cutoff=-1,
    )
    forces = np.zeros((len(job.plan), len(job.supercell), 3))
    result = job.reap(forces)
    assert result.orders == (2,)
    assert result[2].shape == (2, 16, 3, 3)


def test_stage_reporting_is_enabled_by_default(capsys):
    job = ForceConstantCalculation(
        bulk("Si", "diamond", a=5.43),
        order=2,
        supercell=(2, 2, 2),
        cutoff=-1,
    )
    job.evaluate(ZeroCalculator())
    output = capsys.readouterr().out
    assert "Creating reference supercell" in output
    assert "Space group Fd-3m" in output
    assert "cluster equivalence classes" in output
    assert "force calculations required" in output
    assert "Evaluating" in output
    assert "Forces:" in output


def test_stage_reporting_can_be_disabled_completely(capsys):
    job = ForceConstantCalculation(
        bulk("Si", "diamond", a=5.43),
        order=2,
        supercell=(2, 2, 2),
        cutoff=-1,
        verbose=False,
    )
    forces = job.evaluate(ZeroCalculator())
    job.reap(forces)
    assert capsys.readouterr().out == ""


def test_invalid_jax_platform_is_rejected():
    with pytest.raises(ValueError, match="jax_platform"):
        configure_jax("tpu")
