"""Plot phonon bands from SPOSCAR and Phonopy text or HDF5 force constants.

    python plot.py [SPOSCAR] [fc2-phonopy.hdf5|fc2-phonopy.txt]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seekpath
from phonopy import load

ROOT = Path(__file__).resolve().parent

if len(sys.argv) > 3:
    sys.exit(__doc__)

sposcar = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "SPOSCAR"
force_constants = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "fc2-phonopy.hdf5"

phonon = load(
    supercell_filename=sposcar,
    supercell_matrix=np.eye(3, dtype=int),
    primitive_matrix="auto",
    force_constants_filename=force_constants,
    is_compact_fc=False,
    is_nac=False,
)
primitive = phonon.primitive
path = seekpath.get_explicit_k_path(
    (primitive.cell, primitive.scaled_positions, primitive.numbers)
)
seek_cell = np.asarray(path["primitive_lattice"])
qpoints = np.asarray(path["explicit_kpoints_rel"]) @ (
    np.linalg.inv(seek_cell).T @ primitive.cell.T
)
result = phonon.run_qpoints(qpoints)
frequencies = result.frequencies
distances = np.asarray(path["explicit_kpoints_linearcoord"])

fig, axis = plt.subplots(figsize=(9, 5))
for begin, end in path["explicit_segments"]:
    axis.plot(distances[begin:end], frequencies[begin:end], color="#176b87", lw=0.8)
labels = {}
for index, label in enumerate(path["explicit_kpoints_labels"]):
    if label:
        name = "Γ" if label == "GAMMA" else label.replace("_", "")
        labels.setdefault(float(distances[index]), []).append(name)
ticks = list(labels)
axis.set_xticks(ticks, ["|".join(labels[tick]) for tick in ticks])
for tick in ticks:
    axis.axvline(tick, color="0.8", lw=0.5)
axis.axhline(0.0, color="0.5", lw=0.6)
axis.set_ylabel("Frequency (THz)")
fig.tight_layout()
output = ROOT / "phonon-bands.png"
fig.savefig(output, dpi=180)
plt.close(fig)
print(f"primitive atoms: {len(primitive)}")
print(f"wrote {output}")
