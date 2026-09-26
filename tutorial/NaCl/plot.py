"""Compare NaCl phonons from direct fitting and explicit Ewald subtraction."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seekpath
from ase.io import read
from phonopy import Phonopy
from phonopy.file_IO import parse_BORN, read_force_constants_hdf5
from phonopy.structure.atoms import PhonopyAtoms

ROOT = Path(__file__).resolve().parent
unit = read(ROOT / "NaCl_unitcell.xyz")
phonon = Phonopy(
    PhonopyAtoms(
        symbols=unit.get_chemical_symbols(),
        scaled_positions=unit.get_scaled_positions(),
        cell=unit.cell,
    ),
    np.diag([4, 4, 4]),
    primitive_matrix=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
)
phonon.generate_displacements()
frames = read(ROOT / "supercells_with_forces.xyz", index=":")
phonon.forces = [frame.get_forces() for frame in frames]
phonon.produce_force_constants()

primitive = phonon.primitive
path = seekpath.get_explicit_k_path((primitive.cell, primitive.scaled_positions, primitive.numbers))
seek_cell = np.asarray(path["primitive_lattice"])
qpoints = np.asarray(path["explicit_kpoints_rel"]) @ (np.linalg.inv(seek_cell).T @ primitive.cell.T)
distance = np.asarray(path["explicit_kpoints_linearcoord"])

phonon.run_qpoints(qpoints)
reference_without_nac = phonon.qpoints.frequencies.copy()
phonon.nac_params = parse_BORN(primitive, filename=str(ROOT / "BORN"))
phonon.run_qpoints(qpoints)
reference = phonon.qpoints.frequencies.copy()

phonon.force_constants = read_force_constants_hdf5(ROOT / "fc2-direct.hdf5")
phonon.run_qpoints(qpoints)
direct = phonon.qpoints.frequencies.copy()
phonon.force_constants = read_force_constants_hdf5(ROOT / "fc2-corrected.hdf5")
phonon.run_qpoints(qpoints)
corrected = phonon.qpoints.frequencies.copy()

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, sharey=True)
curves = (
    (axes[0], reference_without_nac, "without NAC", "#9bbad0", ":"),
    (axes[0], reference, "with NAC", "#174b73", "-"),
    (axes[1], reference, "phonopy reference + NAC", "#174b73", "-"),
    (axes[1], direct, "direct fit + NAC", "#bc5b31", "--"),
    (axes[1], corrected, "Ewald-corrected fit + NAC", "#399f70", "-"),
)
for axis, values, label, color, style in curves:
    for begin, end in path["explicit_segments"]:
        lines = axis.plot(
            distance[begin:end],
            values[begin:end],
            color=color,
            linestyle=style,
            linewidth=1.2,
        )
        if begin == path["explicit_segments"][0][0]:
            lines[0].set_label(label)

labels = {}
for index, label in enumerate(path["explicit_kpoints_labels"]):
    if label:
        name = "Γ" if label == "GAMMA" else label.replace("_", "")
        labels.setdefault(float(distance[index]), []).append(name)
ticks = list(labels)
for axis in axes:
    axis.set_xticks(ticks, ["|".join(labels[tick]) for tick in ticks])
    axis.set_xlim(distance[0], distance[-1])
    axis.set_ylabel("Frequency (THz)")
    axis.axhline(0, color="0.6", linewidth=0.6)
    for tick in ticks:
        axis.axvline(tick, color="0.85", linewidth=0.5)
    axis.legend(frameon=False, loc="upper right")
axes[0].set_title("Effect of non-analytic correction")
axes[1].set_title("Effect of long-range force subtraction")
fig.tight_layout()
fig.savefig(ROOT / "phonon-bands.png", dpi=180)
plt.close(fig)

print(f"direct vs reference: {np.sqrt(np.mean((direct - reference) ** 2)):.5f} THz")
print(f"corrected vs reference: {np.sqrt(np.mean((corrected - reference) ** 2)):.5f} THz")
print(f"wrote {ROOT / 'phonon-bands.png'}")
