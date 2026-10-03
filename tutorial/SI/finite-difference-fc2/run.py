"""Si FC2 by finite differences.

    python run.py sow   # writes POSCAR-000 .. POSCAR-00N
    python run.py reap  # reads structures/job000/vasprun.xml .. job00N

reap() checks the saved structures against the generated displacement order,
reconstructs FC2, and writes phonopy's text fc2-phonopy.txt file.
"""

import sys
from pathlib import Path

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from mlfcs import ClusterMap, ClusterSpace, FiniteDifference

ROOT = Path(__file__).resolve().parent

cs = ClusterSpace(read(ROOT / "POSCAR"), cutoffs={2: 6}, max_body_orders={2: 2})
fd = FiniteDifference(ClusterMap(cs, read(ROOT / "SPOSCAR")), order=2, disps=0.01)


def sow():
    for i, atoms in enumerate(fd.displacements()):
        write(ROOT / f"POSCAR-{i:03d}", atoms, format="vasp", direct=True, vasp5=True)
    print(f"sowed {fd.n_configurations} structures into {ROOT}")


def reap():
    paths = sorted(
        (ROOT / "structures").glob("job*/vasprun.xml"),
        key=lambda path: int(path.parent.name.removeprefix("job")),
    )
    if len(paths) != fd.n_configurations:
        raise ValueError(
            f"expected {fd.n_configurations} vasprun.xml files in structures/, found {len(paths)}"
        )

    structures = []
    for path in paths:
        atoms = read(path, index=-1, format="vasp-xml")
        forces = atoms.get_forces()
        atoms.calc = SinglePointCalculator(atoms, forces=forces)
        structures.append(atoms)

    projection = fd.reconstruct(structures).enforce_asr(orders=(2,))
    model = projection.force_constants
    report = projection.report(2)
    output = ROOT / "fc2-phonopy.txt"
    model.write(output, fd.cluster_map, format="phonopy", storage="text", order=2)
    print(
        f"reaped {len(structures)} structures into {output}; "
        f"FC2 ASR relative residual {report.relative_before:.3e} "
        f"-> {report.relative_after:.3e}"
    )


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["sow"]:
            sow()
        case ["reap"]:
            reap()
        case _:
            sys.exit(__doc__)
