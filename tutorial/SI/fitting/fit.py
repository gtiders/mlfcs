"""Fit Si force constants, apply ASR, and write Phonopy and phono3py files."""

from __future__ import annotations

import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ase.io import iread, read

from mlfcs import ClusterMap, ClusterSpace, FitSystem, ForceConstants, Supercell
from mlfcs.core.log_error import configure

ROOT = Path(__file__).resolve().parent
CUTOFFS = {2: -7, 3: -6, 4: -3, 5: -2}
MAX_BODY_ORDERS = {2: 2, 3: 3, 4: 4, 5: 3}
SOLVER_RTOL = 1e-8
SOLVER_MAX_STEPS = 10_000
MODEL_FILE = ROOT / "fc-fit.mlfcs"


def fit() -> None:
    space = ClusterSpace(
        read(ROOT / "POSCAR"),
        cutoffs=CUTOFFS,
        max_body_orders=MAX_BODY_ORDERS,
    )
    supercell = Supercell.from_atoms(space.primitive, read(ROOT / "SPOSCAR"))
    mapping = ClusterMap.build(space, supercell)
    for order in space.orders:
        rank = mapping.rank_info(order)
        rank.require_full()

    system = FitSystem.from_atoms(mapping, iread(ROOT / "train.xyz", index=":"))
    parameters = system.solve(rtol=SOLVER_RTOL, max_steps=SOLVER_MAX_STEPS)
    system.force_constants(parameters).save(MODEL_FILE)
    model = ForceConstants.load(MODEL_FILE)
    projection = model.enforce_asr()
    model = projection.force_constants
    model.save(MODEL_FILE)
    parameters = model.parameters()
    print(
        f"post-ASR training force RMSE {system.rmse(parameters):.8g} eV/Å; "
        f"relative force error {system.relative_error(parameters):.8g}"
    )
    model.write(
        ROOT / "fc2-phonopy.hdf5",
        mapping,
        format="phonopy_hdf5",
        order=2,
    )
    model.write(
        ROOT / "fc3-phono3py.hdf5",
        mapping,
        format="phono3py_hdf5",
        order=3,
    )
    print(f"saved ASR-projected model and FC2/FC3 HDF5 files in {ROOT}")


def main() -> None:
    with (ROOT / "fit.log").open("w", encoding="utf-8") as log:
        configure(stream=log)
        try:
            with redirect_stdout(log), redirect_stderr(log):
                fit()
        except BaseException:
            traceback.print_exc(file=log)
            raise


if __name__ == "__main__":
    main()
