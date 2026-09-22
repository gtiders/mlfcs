"""Compute Si FC2 and FC3 by finite differences with the bundled NEP potential."""
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

def _run() -> None:
    primitive = read(ROOT / "POSCAR.vasp")
    reference = build_supercell(primitive, (4, 4, 4))
    write(ROOT / "SPOSCAR", reference, format="vasp", direct=True, sort=False, vasp5=True)
    calculator = CPUNEP(str(MODEL))
    for order, filename in ((2, "fc2"), (3, "fc3")):
        calculation = FiniteDifferenceCalculation(
            primitive,
            order=order,
            reference=reference,
            cutoff=None,
            displacement=0.01,
            symprec=1e-5,
        )
        force_constants = calculation.run(calculator)
        write_force_constants(force_constants, ROOT / f"{filename}-mlfcs.h5", format="hdf5")
        if order == 2:
            write_force_constants(
                force_constants, ROOT / "force_constants.hdf5", format="phonopy_hdf5", order=2,
                primitive=primitive, supercell=reference,
            )
            write_force_constants(
                force_constants, ROOT / "FORCE_CONSTANTS_2ND", format="phonopy", order=2,
                primitive=primitive, supercell=reference,
            )
        else:
            write_force_constants(
                force_constants, ROOT / "fc3.hdf5", format="phono3py_hdf5", order=3,
                primitive=primitive, supercell=reference,
            )
        (ROOT / f"metadata-{filename}.json").write_text(
            json.dumps(force_constants.metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )

def main() -> None:
    with (ROOT / "run.log").open("w", encoding="utf-8") as log:
        handler = logging.StreamHandler(log)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        package_logger = logging.getLogger("mlfcs")
        package_logger.addHandler(handler)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = log, log
        try:
            _run()
        except BaseException:
            traceback.print_exc(file=log)
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            package_logger.removeHandler(handler)
            handler.close()

if __name__ == "__main__":
    main()
