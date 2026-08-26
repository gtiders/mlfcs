from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs import ForceConstantFitter, enforce_rotational_sum_rules
from mlfcs.constraints.translational import project_parameters
from mlfcs.fitting.backends.factory import create_fitting_backend
from mlfcs.fitting.backends.interface import FittingBasisBackend
from mlfcs.fitting.backends.result import LoweringResult
from mlfcs.fitting.backends.taylor.backend import TaylorFittingBackend
from mlfcs.fitting.backends.taylor.features import taylor_axis_derivatives
from mlfcs.fitting.backends.wick.backend import WickFittingBackend
from mlfcs.fitting.backends.wick.features import wick_axis_derivatives
from mlfcs.fitting.constraints import build_joint_constraints


def test_backend_result_objects_are_basis_independent():
    result = LoweringResult(taylor_parameters=np.asarray([1.0]))
    np.testing.assert_array_equal(result.taylor_parameters, [1.0])
    assert FittingBasisBackend is not None


def test_taylor_features_equal_zero_covariance_wick_features():
    displacement = jnp.asarray([[0.2, -0.3, 0.5]])
    coordinates = jnp.asarray([[[0, 1, 2, 0]]])
    covariance = jnp.zeros((3, 3))

    taylor_values = taylor_axis_derivatives(displacement, jnp.empty(0), coordinates, 4)
    wick_values = wick_axis_derivatives(displacement, covariance, coordinates, 4)

    for actual, expected in zip(taylor_values, wick_values, strict=True):
        np.testing.assert_allclose(actual, expected, atol=0.0, rtol=0.0)


def test_taylor_lowering_is_identity():
    parameters = np.asarray([1.0, -2.0, 3.0])
    lowered = TaylorFittingBackend().lower(None, parameters)

    np.testing.assert_array_equal(lowered.taylor_parameters, parameters)
    assert lowered.reference_fc1 is None


def test_backend_factory_is_the_only_basis_name_dispatch():
    assert create_fitting_backend("taylor").name == "taylor"
    assert create_fitting_backend("WICK").name == "wick"
    with pytest.raises(ValueError, match="taylor.*wick"):
        create_fitting_backend("unknown")


def test_fitter_backends_share_result_interface_and_fc2_prediction():
    primitive = Atoms("Ar", cell=np.eye(3) * 4.0, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = primitive.repeat((2, 1, 1))
    structures = []
    for value in (-0.03, -0.01, 0.01, 0.03):
        atoms = reference.copy()
        atoms.positions[0, 0] += value
        forces = np.zeros((len(reference), 3))
        forces[0, 0] = -value
        forces[1, 0] = value
        atoms.calc = SinglePointCalculator(atoms, forces=forces)
        structures.append(atoms)

    results = {}
    for basis in ("taylor", "wick"):
        fitter = ForceConstantFitter(
            primitive,
            reference,
            orders=(2,),
            cutoffs={2: 4.1},
            fitting_basis=basis,
        )
        gram = fitter.prepare_gram(structures, acoustic_sum_rule=False)
        results[basis] = fitter.fit(gram, acoustic_sum_rule=False)

    assert results["taylor"].fitting_basis == "taylor"
    assert results["wick"].fitting_basis == "wick"
    np.testing.assert_allclose(
        results["taylor"].force_constants.sparse[2].tensors,
        results["wick"].force_constants.sparse[2].tensors,
        atol=1e-12,
        rtol=1e-12,
    )

    corrected = {
        basis: enforce_rotational_sum_rules(
            result.force_constants,
            born_huang=True,
            huang=True,
        )
        for basis, result in results.items()
    }
    np.testing.assert_allclose(
        corrected["taylor"].force_constants.sparse[2].tensors,
        corrected["wick"].force_constants.sparse[2].tensors,
        atol=1e-12,
        rtol=1e-12,
    )
    assert corrected["taylor"].acoustic_after < 1e-12
    assert corrected["wick"].acoustic_after < 1e-12


def test_wick_lowering_preserves_per_order_acoustic_null_spaces():
    primitive = Atoms("Ar", cell=np.eye(3) * 4.0, scaled_positions=[[0, 0, 0]], pbc=True)
    reference = primitive.repeat((3, 3, 3))
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2, 3, 4),
        cutoffs={2: 4.1, 3: 4.1, 4: 4.1},
        max_body_orders={2: 2, 3: 2, 4: 2},
        fitting_basis="wick",
    )
    constraints = build_joint_constraints(fitter.calculations, acoustic=True).matrix
    rng = np.random.default_rng(12)
    wick_parameters = project_parameters(
        constraints,
        rng.normal(size=fitter.n_parameters),
        tolerance=1e-10,
    )
    prepared = SimpleNamespace(
        calculations=tuple(fitter.calculations),
        covariance=np.eye(3 * len(reference)) * 0.01,
    )
    taylor_parameters = WickFittingBackend().lower(prepared, wick_parameters).taylor_parameters

    assert np.max(np.abs(constraints @ wick_parameters)) < 1e-10
    assert np.max(np.abs(constraints @ taylor_parameters)) < 1e-10
