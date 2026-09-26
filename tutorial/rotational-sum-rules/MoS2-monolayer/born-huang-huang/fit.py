"""Fit MoS2 FC2 and apply ASR, Born-Huang, and Huang constraints."""

import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ase.io import iread, read

from mlfcs import ClusterMap, ClusterSpace, FitSystem, Supercell
from mlfcs.core.log_error import configure

ROOT = Path(__file__).resolve().parent


def fit():
    space = ClusterSpace(read(ROOT / "primitive.vasp"), cutoffs={2: 8}, max_body_orders={2: 2})
    supercell = Supercell.from_atoms(space.primitive, read(ROOT / "supercell.vasp"))
    mapping = ClusterMap.build(space, supercell)
    system = FitSystem.from_atoms(mapping, iread(ROOT / "training.extxyz", index=":"))
    model = system.force_constants(system.solve(rtol=1e-8, max_steps=10_000))
    model = model.enforce_asr().force_constants
    projection = model.enforce_rotation(born_huang=True, huang=True)
    model = projection.force_constants
    model.save(ROOT / "fc-fit.mlfcs")
    model.write(
        ROOT / "fc2-phonopy.txt", mapping, format="phonopy_text", order=2
    )
    print(
        f"ASR + Born-Huang + Huang: {system.n_structures} frames, "
        f"{system.n_parameters} parameters; relative force error "
        f"{system.relative_error(model.parameters()):.6%}; "
        f"retained rotational rank {projection.retained_rank}"
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
