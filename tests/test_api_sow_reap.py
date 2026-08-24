from typing import ClassVar

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from supercell_helpers import make_supercell

from mlfcs import FiniteDifferenceCalculation
from mlfcs.fitting.jax_runtime import configure_jax


class ZeroCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["forces"]

    def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["forces"] = np.zeros((len(atoms), 3))


def calculation():
    primitive = bulk("Si", "diamond", a=5.43)
    return FiniteDifferenceCalculation(
        primitive,
        order=3,
        reference=make_supercell(primitive, (2, 2, 2))[0],
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
    positional = job.reap(forces).materialize(3)
    mapped = job.reap(
        {index: forces[index] for index in reversed(range(len(structures)))},
    ).materialize(3)
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
    np.testing.assert_array_equal(result.materialize(3), 0.0)
    evaluated = job.evaluate(ZeroCalculator())
    assert evaluated.shape == (len(job.plan), len(job.supercell), 3)
    direct = job.reap(evaluated)
    np.testing.assert_array_equal(direct.materialize(3), 0.0)


def test_second_order_uses_the_same_pipeline():
    primitive = bulk("Si", "diamond", a=5.43)
    job = FiniteDifferenceCalculation(
        primitive,
        order=2,
        reference=make_supercell(primitive, (2, 2, 2))[0],
        cutoff=-1,
    )
    forces = np.zeros((len(job.plan), len(job.supercell), 3))
    result = job.reap(forces)
    assert result.orders == (2,)
    assert result.materialize(2).shape == (2, 16, 3, 3)


def test_stage_reporting_is_enabled_by_default():
    import logging

    primitive = bulk("Si", "diamond", a=5.43)
    job = FiniteDifferenceCalculation(
        primitive,
        order=2,
        reference=make_supercell(primitive, (2, 2, 2))[0],
        cutoff=-1,
    )
    job.evaluate(ZeroCalculator())
    logger = logging.getLogger("mlfcs")
    handlers = [
        handler for handler in logger.handlers if getattr(handler, "_mlfcs_stdout_handler", False)
    ]
    assert logger.level == logging.INFO
    assert len(handlers) == 1
    assert handlers[0].level == logging.NOTSET


def test_stage_reporting_can_be_disabled_completely(capsys):
    primitive = bulk("Si", "diamond", a=5.43)
    job = FiniteDifferenceCalculation(
        primitive,
        order=2,
        reference=make_supercell(primitive, (2, 2, 2))[0],
        cutoff=-1,
    )
    forces = job.evaluate(ZeroCalculator())
    job.reap(forces)
    assert capsys.readouterr().out == ""


def test_invalid_jax_platform_is_rejected():
    with pytest.raises(ValueError, match="jax_platform"):
        configure_jax("tpu")
