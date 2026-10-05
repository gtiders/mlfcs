"""Run the reduced quartic-loop SCPH series and draw temperature-dependent bands."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seekpath

from mlfcs.force_constants import ForceConstants
from mlfcs.phonon import SCPH, Harmonic

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TEMPERATURES = list(range(0, 901, 100))
MESH = np.diag([4, 4, 6])
TOLERANCE_THZ = 1e-9
MIXING = 0.2
MAX_ITERATIONS = 200


class _Tee:
    def __init__(self, terminal, file) -> None:
        self.terminal = terminal
        self.file = file

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.file.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()


def _run() -> None:
    model = ForceConstants.load(ROOT / "fc-fit.mlfcs")
    scph = SCPH(model, MESH)
    results = scph.run_many(
        TEMPERATURES,
        mixing=MIXING,
        tolerance=TOLERANCE_THZ,
        max_iterations=MAX_ITERATIONS,
    )
    for result in results:
        last = result.history[-1]
        print(
            f"{result.temperature:g} K: {result.iterations} iterations, "
            f"converged={result.converged}, change={last.frequency_change_thz:.3e} THz, "
            f"minimum={last.minimum_frequency_thz:.6g} THz, "
            f"minimum_nontranslational={result.minimum_mode_thz:.6g} THz, "
            f"imaginary_modes={result.has_imaginary_modes}"
        )

    primitive = model.cluster_space.primitive_atoms
    path = seekpath.get_explicit_k_path(
        (primitive.cell, primitive.get_scaled_positions(), primitive.numbers),
        symprec=1e-5,
        reference_distance=0.04,
    )
    seek_cell = np.asarray(path["primitive_lattice"])
    qpoints = np.asarray(path["explicit_kpoints_rel"]) @ (
        np.linalg.inv(seek_cell).T @ primitive.cell.T
    )
    distances = np.asarray(path["explicit_kpoints_linearcoord"])
    bands = np.stack([Harmonic(result.fc2).frequencies(qpoints) for result in results])
    labels: dict[float, list[str]] = {}
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    for index, label in enumerate(path["explicit_kpoints_labels"]):
        if label:
            position = float(distances[index])
            name = label.replace("GAMMA", "Γ").replace("SIGMA", "Σ").replace("DELTA", "Δ")
            name = name.replace("_", "").translate(subscripts)
            if name not in labels.setdefault(position, []):
                labels[position].append(name)

    fig, axis = plt.subplots(figsize=(11, 5.5))
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(results)))
    for result, color, frequencies in zip(results, colors, bands, strict=True):
        for begin, end in path["explicit_segments"]:
            axis.plot(distances[begin:end], frequencies[begin:end], color=color, lw=0.65)
    for position in labels:
        axis.axvline(position, color="0.8", lw=0.5)
    axis.axhline(0.0, color="0.5", lw=0.6)
    axis.set_xticks(list(labels), ["|".join(names) for names in labels.values()])
    axis.tick_params(axis="x", labelrotation=35, labelsize=9)
    axis.set_xlabel("seekpath wave vector")
    axis.set_ylabel("Frequency (THz)")
    axis.set_title("K4As4Pt2 quartic-loop SCPH")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(TEMPERATURES[0], TEMPERATURES[-1]), cmap="viridis"
        ),
        ax=axis,
    )
    colorbar.set_label("Temperature (K)")
    fig.tight_layout()
    fig.savefig(HERE / "temperature-bands.png", dpi=180)
    plt.close(fig)

    payload = {
        "mesh": MESH.tolist(),
        "full_qpoints": scph.stars.grid.size,
        "irreducible_qpoints": len(scph.stars.representatives),
        "statistics": "quantum",
        "mixing": MIXING,
        "convergence_tolerance_thz": TOLERANCE_THZ,
        "max_iterations": MAX_ITERATIONS,
        "solve_order": "descending_temperature",
        "temperatures": [
            {
                "kelvin": result.temperature,
                "converged": result.converged,
                "iterations": result.iterations,
                "last_change_thz": result.history[-1].frequency_change_thz,
                "minimum_mesh_frequency_thz": float(np.min(result.frequencies)),
                "minimum_nontranslational_frequency_thz": result.minimum_mode_thz,
                "has_imaginary_modes": result.has_imaginary_modes,
                "maximum_mesh_frequency_thz": float(np.max(result.frequencies)),
                "history": [
                    {
                        "iteration": step.index,
                        "frequency_change_thz": step.frequency_change_thz,
                        "minimum_frequency_thz": step.minimum_frequency_thz,
                    }
                    for step in result.history
                ],
            }
            for result in results
        ],
        "band_path": {
            "qpoints": qpoints.tolist(),
            "distances": distances.tolist(),
            "labels": path["explicit_kpoints_labels"],
            "segments": path["explicit_segments"],
            "frequencies_thz": bands.tolist(),
        },
    }
    (HERE / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not all(result.converged for result in results):
        raise RuntimeError("at least one temperature did not meet the stated SCPH tolerance")


def main() -> None:
    with (HERE / "run.log").open("w", encoding="utf-8") as file:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, file), _Tee(stderr, file)
        try:
            _run()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == "__main__":
    main()
