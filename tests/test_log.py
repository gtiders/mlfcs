"""Focused tests for package-owned logging configuration."""

from __future__ import annotations

import logging
from io import StringIO

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from mlfcs import ClusterMap, ClusterSpace, FitSystem, PrimitiveCell, Supercell
from mlfcs.core.log_error import AliasingError, configure, log_error


def test_logging_can_write_to_console_and_a_file_without_root_configuration(
    tmp_path,
) -> None:
    package = logging.getLogger("mlfcs")
    root = logging.getLogger()
    old_handlers = tuple(package.handlers)
    old_level = package.level
    old_propagate = package.propagate
    old_streams = {
        handler: handler.stream for handler in old_handlers if hasattr(handler, "stream")
    }
    root_handlers = tuple(root.handlers)
    stream = StringIO()
    path = tmp_path / "run.log"
    try:
        configured = configure(
            level=logging.DEBUG,
            stream=stream,
            file=path,
            file_mode="w",
            format="%(levelname)s %(name)s: %(message)s",
        )
        configured.info("fit summary: structures=12 parameters=34")
        configured.debug("solver detail")

        assert "fit summary: structures=12 parameters=34" in stream.getvalue()
        assert "solver detail" in stream.getvalue()
        contents = path.read_text(encoding="utf-8")
        assert "fit summary: structures=12 parameters=34" in contents
        assert "solver detail" in contents
        assert tuple(root.handlers) == root_handlers
        assert configured.propagate is False
    finally:
        for handler in tuple(package.handlers):
            if handler not in old_handlers:
                package.removeHandler(handler)
                handler.close()
        for handler, previous_stream in old_streams.items():
            if hasattr(handler, "setStream") and handler.stream is not previous_stream:
                handler.setStream(previous_stream)
        package.setLevel(old_level)
        package.propagate = old_propagate


def test_logging_rejects_invalid_file_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="file_mode"):
        configure(file=tmp_path / "run.log", file_mode="r")


def test_domain_errors_log_only_when_explicitly_reported() -> None:
    package = logging.getLogger("mlfcs")
    old_handlers = tuple(package.handlers)
    old_level = package.level
    old_propagate = package.propagate
    old_streams = {
        handler: handler.stream for handler in old_handlers if hasattr(handler, "stream")
    }
    stream = StringIO()
    try:
        configure(stream=stream)
        error = AliasingError("rank 31/32")
        assert not stream.getvalue()
        try:
            raise error
        except AliasingError as caught:
            caught.log(message="Cluster map is not identifiable")

        output = stream.getvalue()
        assert "Cluster map is not identifiable" in output
        assert "AliasingError: rank 31/32" in output
        assert "Traceback (most recent call last)" in output
        stream.seek(0)
        stream.truncate(0)
        log_error(ValueError("bad input"), message="Input was rejected")
        assert "Input was rejected" in stream.getvalue()
        assert "Traceback" not in stream.getvalue()
    finally:
        for handler in tuple(package.handlers):
            if handler not in old_handlers:
                package.removeHandler(handler)
                handler.close()
        for handler, previous_stream in old_streams.items():
            if hasattr(handler, "setStream") and handler.stream is not previous_stream:
                handler.setStream(previous_stream)
        package.setLevel(old_level)
        package.propagate = old_propagate


def test_fit_logs_the_complete_model_solver_and_quality_summary() -> None:
    package = logging.getLogger("mlfcs")
    old_handlers = tuple(package.handlers)
    old_level = package.level
    old_propagate = package.propagate
    old_streams = {
        handler: handler.stream for handler in old_handlers if hasattr(handler, "stream")
    }
    stream = StringIO()
    try:
        configure(stream=stream)
        atoms = bulk("Ar", "sc", a=1.0)
        primitive = PrimitiveCell.from_atoms(atoms)
        space = ClusterSpace(
            atoms,
            cutoffs={2: -1},
            max_body_orders={2: 1},
        )
        supercell = Supercell.from_atoms(primitive, atoms)
        mapping = ClusterMap.build(space, supercell)
        structures = []
        for displacement in np.eye(3) * 0.01:
            sample = atoms.copy()
            sample.positions += displacement
            sample.calc = SinglePointCalculator(sample, forces=(-2.0 * displacement)[None, :])
            structures.append(sample)
        system = FitSystem.from_atoms(mapping, structures)
        system.solve(rtol=1e-12)

        output = stream.getvalue()
        for required in (
            "requested cutoff 1 neighbor shells",
            "cutoff_angstrom",
            "max_body_order",
            "orbits",
            "parameters=1",
            "structures=3 equations=9",
            "column-scaled-MINRES",
            "training_force_rmse=",
            "training_relative_force_error=",
            "fit_system_fingerprint=",
            "force_constants_fingerprint=",
        ):
            assert required in output
    finally:
        for handler in tuple(package.handlers):
            if handler not in old_handlers:
                package.removeHandler(handler)
                handler.close()
        for handler, previous_stream in old_streams.items():
            if hasattr(handler, "setStream") and handler.stream is not previous_stream:
                handler.setStream(previous_stream)
        package.setLevel(old_level)
        package.propagate = old_propagate
