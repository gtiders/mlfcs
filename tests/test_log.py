"""Printed progress summaries for every task in the library."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs import ClusterMap, ClusterSpace, FiniteDifference, FitSystem, ForceConstants
from mlfcs.dataset import ForceDataset
from mlfcs.fitting.design import ForceDesign
from mlfcs.phonon import SCPH, Harmonic


@pytest.mark.parametrize(
    "representation,algorithm", [("normal", "MINRES"), ("raw", "least_squares")]
)
def test_fit_prints_progress_quality_and_returns_a_finite_model(representation, algorithm):
    """Verify a fit prints its stages and keeps the fitted model finite."""
    output = StringIO()
    with redirect_stdout(output):
        space = ClusterSpace(bulk("Ar", "sc", a=1.0), cutoffs={2: 0.1})
        mapping = ClusterMap(space, space.primitive_atoms)
        design = ForceDesign(mapping)
        structures = []
        for index in range(12):
            displacement = np.array([[0.01, 0.02, 0.03]]) * (index + 1)
            atoms = mapping.supercell_atoms
            atoms.positions += displacement
            atoms.calc = SinglePointCalculator(
                atoms, forces=(design.matrix(displacement) @ [2.0]).reshape(-1, 3)
            )
            structures.append(atoms)
        consumed = []

        def samples():
            """Provide the prepared structures one at a time."""
            for index, atoms in enumerate(structures):
                consumed.append(index)
                yield atoms

        system = FitSystem(ForceDataset(mapping, samples()), representation=representation)
        model = system.solve()
    assert consumed == list(range(12))
    assert np.all(np.isfinite(model.parameters()))
    text = output.getvalue()
    for required in (
        "Build ClusterSpace:",
        "Symmetry:",
        "Total: orbits ",
        "Mapping complete",
        "Force design ready:",
        "Fit system:",
        "Fit accumulation: structures 1,",
        "Fit accumulation: structures 10,",
        "Fit equations:",
        "Fit solve:",
        "Fit complete: RMSE ",
        f"Solver: {algorithm.replace('_', ' ')}",
    ):
        assert required in text


def test_task_summaries_cover_difference_projection_phonons_and_output(tmp_path):
    """Verify every task prints a summary for its inputs and results."""
    output = StringIO()
    with redirect_stdout(output):
        atoms = bulk("Ar", "sc", a=1.0)
        space = ClusterSpace(atoms, cutoffs={2: 1.1, 4: 0.1}, asr=True)
        supercell = atoms.repeat((2, 2, 2))
        supercell_matrix = 2 * np.eye(3, dtype=np.int64)
        coefficients = []
        for orbit in space.orbits[space.block(2).orbits]:
            tensor = (
                6.0 if orbit.representative.sites[0] == orbit.representative.sites[1] else -1.0
            ) * np.eye(3)
            coefficients.extend(
                np.linalg.lstsq(orbit.component_basis, tensor.reshape(-1), rcond=None)[0]
            )
        model = ForceConstants(
            space,
            {
                2: coefficients,
                4: np.zeros(space.block(4).parameters.stop - space.block(4).parameters.start),
            },
        )
        model.enforce_rotation()
        Harmonic(model).mesh((2, 2, 2))
        result = SCPH(model, (2, 2, 2)).run(100, max_iterations=2)
        assert result.converged
        model.write(
            tmp_path / "FORCE_CONSTANTS",
            ClusterMap(space, supercell, supercell_matrix=supercell_matrix),
            format="phonopy",
            order=2,
        )
        ForceConstants.load(model.save(tmp_path / "fc.mlfcs"))
        onsite = ClusterSpace(atoms, cutoffs={2: 0.1})
        mapping = ClusterMap(onsite, atoms)
        difference = FiniteDifference(mapping, order=2)

        class Spring(Calculator):
            """On-site ASE force calculator with stiffness 2 eV/angstrom**2."""

            implemented_properties = ("forces",)

            def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
                """Store forces -2*r of shape (n_atoms, 3) in eV/angstrom."""
                super().calculate(atoms, properties, system_changes)
                self.results = {"forces": -2.0 * atoms.positions}

        samples = []
        calculator = Spring()
        for sample in difference.sow():
            calculator.calculate(sample, properties=["forces"], system_changes=all_changes)
            sample.calc = SinglePointCalculator(sample, forces=calculator.results["forces"])
            samples.append(sample)
        difference.reap(ForceDataset(mapping, samples))
    text = output.getvalue()
    for required in (
        "Rotation projection",
        "Harmonic mesh:",
        "SCPH complete:",
        "Saved model ",
        "Loaded model ",
        "Exported ",
        "Finite difference reconstruction:",
    ):
        assert required in text
