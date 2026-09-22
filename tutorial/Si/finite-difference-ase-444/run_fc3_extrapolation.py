"""Compute Si FC3 by five-step zero-step finite-difference extrapolation."""
from __future__ import annotations
import json
import logging
import sys
import traceback
from pathlib import Path
from ase.io import read, write
from calorine.calculators import CPUNEP
from mlfcs import FiniteDifferenceCalculation, build_supercell, write_force_constants
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "Si_2022_NEP3_5body.txt"
def run() -> None:
    primitive = read(ROOT / "POSCAR.vasp")
    reference = build_supercell(primitive, (4, 4, 4))
    write(ROOT / "SPOSCAR-extrapolated-fc3", reference, format="vasp", direct=True, sort=False, vasp5=True)
    calculation = FiniteDifferenceCalculation(primitive, order=3, reference=reference, cutoff=None, displacement=0.020, symprec=1e-5)
    force_constants = calculation.run(CPUNEP(str(MODEL)), derivative_backend="extrapolate", extrapolation_spacing=0.005, extrapolation_side_steps=2, extrapolation_degree=2)
    write_force_constants(force_constants, ROOT / "fc3-extrapolated-mlfcs.h5", format="hdf5")
    write_force_constants(force_constants, ROOT / "fc3-extrapolated.hdf5", format="phono3py_hdf5", order=3, primitive=primitive, supercell=reference)
    (ROOT / "metadata-fc3-extrapolated.json").write_text(json.dumps(force_constants.metadata, indent=2, default=str) + "\n", encoding="utf-8")
def main() -> None:
    with (ROOT / "fc3-extrapolation.log").open("w", encoding="utf-8") as log:
        handler = logging.StreamHandler(log); handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s")); package_logger = logging.getLogger("mlfcs"); package_logger.addHandler(handler)
        stdout, stderr = sys.stdout, sys.stderr; sys.stdout = log; sys.stderr = log
        try: run()
        except BaseException:
            traceback.print_exc(file=log); raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr; package_logger.removeHandler(handler); handler.close()
if __name__ == "__main__": main()
