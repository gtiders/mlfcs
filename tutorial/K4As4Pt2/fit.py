"""Fit K4As4Pt2 FC2, FC3, and FC4 and apply ASR."""

import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ase.io import iread, read
from ase.units import Bohr

from mlfcs import ClusterMap, ClusterSpace, FitSystem

ROOT = Path(__file__).resolve().parent
CUTOFFS = {2: 6.5, 3: 12 * Bohr, 4: 8 * Bohr}
MAX_BODY_ORDERS = {2: 2, 3: 3, 4: 3}


def fit():
    space = ClusterSpace(
        read(ROOT / "primitive.vasp"),
        cutoffs=CUTOFFS,
        max_body_orders=MAX_BODY_ORDERS,
    )
    mapping = ClusterMap(space, read(ROOT / "supercell.vasp"))
    system = FitSystem(mapping, iread(ROOT / "train.extxyz", index=":"))
    model = system.solve(rtol=1e-8, maxiter=10_000)
    model = model.enforce_asr().force_constants
    model.save(ROOT / "fc-fit.mlfcs")
    model.write(
        ROOT / "fc2-phonopy.txt", mapping, format="phonopy", storage="text", order=2
    )
    model.write(ROOT / "fc3-shengbte.txt", mapping, format="shengbte", order=3)
    model.write(ROOT / "fc4-shengbte.txt", mapping, format="shengbte", order=4)
    print(
        f"K4As4Pt2: {system.n_structures} structures, {system.n_parameters} parameters; "
        f"relative force error {system.relative_error(model.parameters()):.6%} after ASR"
    )


def main():
    with (ROOT / "fit.log").open("w", encoding="utf-8") as log:
        try:
            with redirect_stdout(log), redirect_stderr(log):
                fit()
        except BaseException:
            traceback.print_exc(file=log)
            raise


if __name__ == "__main__":
    main()
