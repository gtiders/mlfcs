"""Package stdout logging and high-level scientific task summaries."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs import ClusterMap, ClusterSpace, FiniteDifference, FitSystem, ForceConstants
from mlfcs.core.log import configure, get_logger
from mlfcs.fitting.design import ForceDesign
from mlfcs.reciprocal import SCPH, Harmonic


@pytest.fixture
def package_log():
    """Isolate package handlers/level and restore them even after a failed assertion."""
    package = logging.getLogger("mlfcs")
    handlers, level, propagate = package.handlers[:], package.level, package.propagate
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


def test_configuration_is_idempotent_and_does_not_touch_root_or_user_handlers(package_log):
    root = logging.getLogger()
    root_state = root.handlers[:], root.level
    user = logging.NullHandler()
    package_log.addHandler(user)
    original_handlers = package_log.handlers[:]
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
    assert re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d DEBUG mlfcs.test: ", text)
    assert user in package_log.handlers
    assert package_log.handlers == original_handlers
    assert (root.handlers, root.level) == root_state
    assert package_log.propagate is False
    with pytest.raises(TypeError):
        configure(stream=output)


def test_current_stdout_survives_closed_redirect_and_standard_exception_logging(
    package_log, tmp_path
):
    log = get_logger("test")
    path = tmp_path / "captured.log"
    with path.open("w") as file, redirect_stdout(file):
        log.info("inside-marker")
        error = ValueError("caught-marker")
        assert "caught-marker" not in path.read_text()
        try:
            raise error
        except ValueError:
            log.exception("task failed")
    output = StringIO()
    with redirect_stdout(output):
        log.info("after-marker")
    assert "after-marker" in output.getvalue()
    text = path.read_text()
    assert "inside-marker" in text and "task failed" in text
    assert "Traceback (most recent call last)" in text and "ValueError: caught-marker" in text
    assert "after-marker" not in text


def test_bash_redirect_collects_stdout_stderr_traceback_and_keeps_exit_code(tmp_path):
    path = tmp_path / "task.log"
    code = """import sys
from mlfcs.core.log import get_logger
print("stdout-marker")
print("stderr-marker", file=sys.stderr)
get_logger("task").info("package-marker")
try:
    raise ValueError("traceback-marker")
except ValueError:
    get_logger("task").exception("task failed")
sys.exit(3)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    completed = subprocess.run(
        ["bash", "-c", '"$1" -c "$2" > "$3" 2>&1', "logging-task", sys.executable, code, str(path)],
        env=env,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 3
    assert completed.stdout == completed.stderr == b""
    text = path.read_text()
    for marker in (
        "stdout-marker",
        "stderr-marker",
        "package-marker",
        "Traceback",
        "traceback-marker",
    ):
        assert marker in text


@pytest.mark.parametrize("representation,algorithm", [("normal", "MINRES"), ("raw", "LSMR")])
def test_fit_logs_progress_quality_and_preserves_results(package_log, representation, algorithm):
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
            for index, atoms in enumerate(structures):
                consumed.append(index)
                yield atoms

        system = FitSystem(mapping, samples(), representation=representation)
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
        "stop_code=",
        "training_force_rmse=",
        "training_relative_force_error=",
        "fit_system_fingerprint=",
        "force_constants_fingerprint=",
    ):
        assert required in text


def test_task_summaries_cover_difference_projection_phonons_and_output(package_log, tmp_path):
    output = StringIO()
    with redirect_stdout(output):
        atoms = bulk("Ar", "sc", a=1.0)
        space = ClusterSpace(atoms, cutoffs={2: 1.1, 4: 0.1})
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
        model.enforce_asr()
        model.enforce_rotation()
        Harmonic(model).mesh((2, 2, 2))
        result = SCPH(model, (2, 2, 2)).run(100, max_iterations=2)
        assert result.converged
        model.write(tmp_path / "fc.tdep", format="tdep", order=2)
        ForceConstants.load(model.save(tmp_path / "fc.mlfcs"))
        onsite = ClusterSpace(atoms, cutoffs={2: 0.1})
        mapping = ClusterMap(onsite, atoms)
        difference = FiniteDifference(mapping, order=2)

        # Keep the force oracle independent of material-specific potentials.
        class Spring(Calculator):
            implemented_properties = ["forces"]

            def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                self.results = {"forces": -2.0 * atoms.positions}

        difference.reconstruct(difference.evaluate(Spring()))
    text = output.getvalue()
    for required in (
        "ASR projection started",
        "ASR projection complete",
        "Rotation projection complete",
        "Harmonic mesh complete",
        "SCPH run complete",
        "Native save complete",
        "Native load complete",
        "Export complete",
        "Finite-difference evaluation complete",
        "Finite-difference reconstruction complete",
    ):
        assert required in text


def test_tutorials_capture_their_own_logs_without_package_handlers():
    root = Path(__file__).parents[1] / "tutorial"
    scripts = [path for path in root.rglob("fit.py") if '"fit.log"' in path.read_text()]
    assert scripts
    for path in scripts:
        source = path.read_text()
        assert '.open("w", encoding="utf-8")' in source
        assert "traceback.print_exc" in source
        assert "redirect_stdout" in source or "sys.stdout, sys.stderr = _Tee" in source
        assert "configure(stream=" not in source
        assert "addHandler" not in source
