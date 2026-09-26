"""Fit the 300 K Ba8Ga16Ge30 effective FC2+FC3 model and apply ASR."""

import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ase.io import iread, read

from mlfcs import ClusterMap, ClusterSpace, FitSystem, Supercell
from mlfcs.core.log_error import configure

ROOT = Path(__file__).resolve().parent
TEMPERATURE_K = 300
CUTOFFS = {2: 5.4, 3: 4.35}
MAX_BODY_ORDERS = {2: 2, 3: 2}
SYMPREC = 1e-4


def fit():
    space = ClusterSpace(
        read(ROOT / "primitive.vasp"),
        cutoffs=CUTOFFS,
        max_body_orders=MAX_BODY_ORDERS,
        symprec=SYMPREC,
    )
    supercell = Supercell.from_atoms(space.primitive, read(ROOT / "supercell.vasp"))
    mapping = ClusterMap.build(space, supercell)
    system = FitSystem.from_atoms(mapping, iread(ROOT / "nve.extxyz", index=":"))
    model = system.force_constants(system.solve(rtol=1e-8, max_steps=10_000))
    model = model.enforce_asr().force_constants
    model.save(ROOT / "fc-fit.mlfcs")
    model.write(
        ROOT / "fc2-phonopy.txt", mapping, format="phonopy_text", order=2
    )
    model.write(ROOT / "fc3-shengbte.txt", mapping, format="shengbte", order=3)
    print(
        f"Ba8Ga16Ge30 at {TEMPERATURE_K} K: {system.n_structures} frames, "
        f"{system.n_parameters} parameters; relative force error "
        f"{system.relative_error(model.parameters()):.6%} after ASR"
    )


def main():
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
