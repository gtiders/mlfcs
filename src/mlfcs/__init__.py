"""ASE-first anharmonic force-constant tools."""

import logging
import sys
from importlib import import_module


def _configure_package_logger() -> None:
    logger = logging.getLogger("mlfcs")
    if not any(getattr(handler, "_mlfcs_stdout_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler._mlfcs_stdout_handler = True
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


_configure_package_logger()

from mlfcs.calculators.ase import MLFCSCalculator
from mlfcs.constraints.rotational import enforce_rotational_sum_rules
from mlfcs.finite_difference.calculation import FiniteDifferenceCalculation
from mlfcs.force_constants.realization import realize_force_constants
from mlfcs.force_constants.representation import ForceConstants
from mlfcs.io.hdf5 import read_hdf5
from mlfcs.io.write import write_force_constants
from mlfcs.physics.scph.solver import LoopSCPH
from mlfcs.sampling import perturb_structures
from mlfcs.structure.supercell import build_supercell

__all__ = [
    "SSCHA",
    "FiniteDifferenceCalculation",
    "ForceConstantFitter",
    "ForceConstants",
    "LoopSCPH",
    "MLFCSCalculator",
    "build_supercell",
    "enforce_rotational_sum_rules",
    "perturb_structures",
    "read_hdf5",
    "realize_force_constants",
    "write_force_constants",
]

__version__ = "4.0.0a6"


def __getattr__(name: str):
    """Load fitting and SSCHA APIs only when explicitly requested."""
    if name == "ForceConstantFitter":
        return getattr(import_module("mlfcs.fitting"), name)
    if name == "SSCHA":
        return getattr(import_module("mlfcs.physics.sscha.solver"), name)
    raise AttributeError(name)
