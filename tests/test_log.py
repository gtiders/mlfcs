"""Numerical behavior of log."""

from __future__ import annotations

import logging
import re
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
from mlfcs.foundation.log import configure, get_logger
from mlfcs.phonon import SCPH, Harmonic


@pytest.fixture
def package_log():
    """Isolate package handlers/level and restore them even after a failed assertion."""
    package = logging.getLogger("mlfcs")
    handlers, level, propagate = (package.handlers[:], package.level, package.propagate)
    package.handlers = []
    configure()
    try:
        yield package
    finally:
        for handler in package.handlers:
            if handler not in handlers:
                handler.close()
        package.handlers = handlers
        package.setLevel(level)
        package.propagate = propagate


def test_logging_configuration_sets_levels_without_duplicate_output(package_log):
    """Verify logging configuration sets levels without duplicate output."""
    output = StringIO()
    with redirect_stdout(output):
        configure(level=logging.DEBUG)
        configure(level=logging.DEBUG)
        get_logger("test").debug("once-marker")
        configure(level=logging.WARNING)
        get_logger("test").info("hidden-marker")
        get_logger("test").warning("visible-marker")
    text = output.getvalue()
    assert text.count("once-marker") == 1
    assert "hidden-marker" not in text
    assert "visible-marker" in text
    assert re.match("\\d{4}-\\d\\d-\\d\\d \\d\\d:\\d\\d:\\d\\d DEBUG mlfcs.test: ", text)


def test_logging_follows_current_stdout(package_log, tmp_path):
    """Verify logging follows current stdout."""
    log = get_logger("test")
    path = tmp_path / "captured.log"
    with path.open("w") as file, redirect_stdout(file):
        log.info("inside-marker")
    output = StringIO()
    with redirect_stdout(output):
        log.info("after-marker")
    assert "inside-marker" in path.read_text()
    assert "after-marker" in output.getvalue()
    assert "after-marker" not in path.read_text()


@pytest.mark.parametrize(
    "representation,algorithm", [("normal", "MINRES"), ("raw", "least_squares")]
)
def test_fit_logs_progress_quality_and_preserves_results(package_log, representation, algorithm):
    """Verify fit logs progress quality and preserves results."""
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
            """Provide samples."""
            for index, atoms in enumerate(structures):
                consumed.append(index)
                yield atoms

        system = FitSystem(ForceDataset(mapping, samples()), representation=representation)
        configure(level=logging.DEBUG)
        observed = system.solve()
        configure(level=logging.WARNING)
        quiet = system.solve()
    np.testing.assert_array_equal(observed.parameters(), quiet.parameters())
    assert consumed == list(range(12))
    text = output.getvalue()
    for required in (
        "Cluster-space construction started",
        "Primitive symmetry ready",
        "Candidates ready",
        "Mapping complete",
        "Folded rank complete",
        "Force-design compilation complete",
        "Fit-system construction started",
        "Fit-system accumulation: structures=1 ",
        "Fit-system accumulation: structures=10 ",
        "Fit-system complete",
        "Fit solve started",
        f"algorithm={algorithm}",
        "stop_code=" if representation == "normal" else "rank=",
        "training_force_rmse=",
        "training_relative_force_error=",
    ):
        assert required in text


def test_task_summaries_cover_difference_projection_phonons_and_output(package_log, tmp_path):
    """Verify task summaries cover difference projection phonons and output."""
    output = StringIO()
    with redirect_stdout(output):
        atoms = bulk("Ar", "sc", a=1.0)
        space = ClusterSpace(atoms, cutoffs={2: 1.1, 4: 0.1}, asr=True)
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
            tmp_path / "FORCE_CONSTANTS", ClusterMap(space, atoms), format="phonopy", order=2
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
        for sample in difference.displacements():
            calculator.calculate(sample, properties=["forces"], system_changes=all_changes)
            sample.calc = SinglePointCalculator(sample, forces=calculator.results["forces"])
            samples.append(sample)
        difference.reconstruct(ForceDataset(mapping, samples))
    text = output.getvalue()
    for required in (
        "Rotation projection complete",
        "Harmonic mesh complete",
        "SCPH run complete",
        "Native save complete",
        "Native load complete",
        "Export complete",
        "Finite-difference reconstruction complete",
    ):
        assert required in text
