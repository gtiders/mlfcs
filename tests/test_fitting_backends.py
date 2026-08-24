from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs import ForceConstantFitter
from mlfcs.fitting.backends.factory import create_fitting_backend
from mlfcs.fitting.backends.interface import FittingBasisBackend
from mlfcs.fitting.backends.result import BasisDiagnostics, BasisLoweringResult
from mlfcs.fitting.backends.taylor.backend import TaylorFittingBackend
from mlfcs.fitting.backends.taylor.features import taylor_axis_derivatives
from mlfcs.fitting.backends.wick.features import wick_axis_derivatives


def test_backend_result_objects_are_basis_independent():
    diagnostics = BasisDiagnostics(details={"backend": "test"})
    result = BasisLoweringResult(taylor_parameters=[1.0], diagnostics=diagnostics)

    assert result.diagnostics.details == {"backend": "test"}
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
    assert lowered.diagnostics.reference_fc1 is None


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
            verbose=False,
        )
        results[basis] = fitter.fit(
            structures,
            validation_split=0.0,
            acoustic_sum_rule=False,
        )

    assert results["taylor"].fitting_basis == "taylor"
    assert results["wick"].fitting_basis == "wick"
    assert not hasattr(results["taylor"].basis_diagnostics, "covariance")
    assert results["wick"].basis_diagnostics.covariance is not None
    np.testing.assert_allclose(
        results["taylor"].force_constants.sparse[2].tensors,
        results["wick"].force_constants.sparse[2].tensors,
        atol=1e-12,
        rtol=1e-12,
    )
