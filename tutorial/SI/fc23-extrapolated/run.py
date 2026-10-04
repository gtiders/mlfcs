"""Extrapolate Si FC2 and FC3 with the Erhart-Albe Si-II potential."""

from pathlib import Path

from ase.calculators.tersoff import Tersoff
from ase.io import read

from mlfcs import ClusterMap, ClusterSpace, FiniteDifference

ROOT = Path(__file__).resolve().parent
DISPS = (0.005, 0.01, 0.015)
POTENTIAL_FILE = ROOT / "Erhart-Albe-Si-II.tersoff"
# 正的绝对截断(Å),对应远程负壳语义 {2: -7, 3: -6} 的解析值
CUTOFFS = {2: 7.418817376, 3: 6.900754996}
MAX_BODY_ORDERS = {2: 2, 3: 3}

space = ClusterSpace(
    read(ROOT / "POSCAR"),
    cutoffs=CUTOFFS,
    max_body_orders=MAX_BODY_ORDERS,
)
mapping = ClusterMap(space, read(ROOT / "SPOSCAR"))
calculator = Tersoff.from_lammps(POTENTIAL_FILE)

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
            storage="hdf5",
            order=2,
        )
    else:
        model.save(ROOT / "fc3-extrapolated.mlfcs")
        model.write(
            ROOT / "fc3-extrapolated.hdf5",
            mapping,
            format="phono3py",
            storage="hdf5",
            order=3,
        )
    print(
        f"FC{order}: steps={DISPS} Å, configurations={calculation.n_configurations}, "
        f"parameters={len(model.coefficients[order])}, "
        f"ASR residual={report.relative_before:.3e} -> {report.relative_after:.3e}"
    )
