"""Extrapolate Si FC2 and FC3 from several finite-difference steps."""

from pathlib import Path

from ase.io import read
from calorine.calculators import CPUNEP

from mlfcs import ClusterMap, ClusterSpace, FiniteDifference, Supercell

ROOT = Path(__file__).resolve().parent
DISPS = (0.005, 0.01, 0.015)
CUTOFFS = {2: -7, 3: -6}
MAX_BODY_ORDERS = {2: 2, 3: 3}

space = ClusterSpace(
    read(ROOT / "POSCAR"),
    cutoffs=CUTOFFS,
    max_body_orders=MAX_BODY_ORDERS,
)
supercell = Supercell.from_atoms(space.primitive, read(ROOT / "SPOSCAR"))
mapping = ClusterMap.build(space, supercell)
calculator = CPUNEP(str(ROOT / "Si_2022_NEP3_5body.txt"))

for order in (2, 3):
    calculation = FiniteDifference(mapping, order=order, disps=DISPS)
    raw = calculation.reconstruct(calculation.evaluate(calculator))
    projection = raw.enforce_asr(orders=(order,))
    model = projection.force_constants
    report = projection.report(order)
    if order == 2:
        model.save(ROOT / "fc2-extrapolated.mlfcs")
        model.write(
            ROOT / "fc2-extrapolated.hdf5",
            mapping,
            format="phonopy",
            order=2,
            storage="hdf5",
        )
    else:
        model.save(ROOT / "fc3-extrapolated.mlfcs")
        model.write(
            ROOT / "fc3-extrapolated.hdf5",
            mapping,
            format="phono3py",
            order=3,
        )
    print(
        f"FC{order}: steps={DISPS} Å, configurations={calculation.n_configurations}, "
        f"parameters={len(model.coefficients[order])}, "
        f"ASR residual={report.relative_before:.3e} -> {report.relative_after:.3e}"
    )
