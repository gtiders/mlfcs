#!/usr/bin/env python3
"""Run reproducible interaction-algebra correctness and timing studies."""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

from ase.build import bulk
from ase.io import read
from finite_pair_prototype import validate_finite_pair_case, validate_periodic_completion
from primitive_prototype import validate_primitive_case

from mlfcs import ForceConstantFitter, build_supercell
from mlfcs.interactions.space import ReferenceFrame

ROOT = Path(__file__).resolve().parents[2]


def _primitive_cases(extended: bool, high_order: bool):
    silicon_orders = [(2, 5.4, 2), (3, 5.4, 3), (4, 4.6, 3)]
    if high_order:
        silicon_orders.extend(((5, 4.6, 3), (6, 4.6, 3)))
    cases = [
        ("Si", bulk("Si", "diamond", a=5.43), 1e-5, tuple(silicon_orders)),
    ]
    if extended:
        cases.extend(
            [
                (
                    "SnSe",
                    read(ROOT / "tutorial/SnSe/joint-fc234/primitive.vasp"),
                    1e-4,
                    ((2, 8.0, 2), (3, 6.5, 3), (4, 4.5, 3)),
                ),
                (
                    "Ba8Ga16Ge30",
                    read(ROOT / "tutorial/Ba8Ga16Ge30/T300K/primitive.vasp"),
                    1e-4,
                    ((2, 5.4, 2), (3, 4.35, 2), (4, 4.35, 2)),
                ),
            ]
        )
    return cases


def run(*, extended: bool, high_order: bool = False) -> dict[str, object]:
    primitive_results = []
    for name, primitive, symprec, specifications in _primitive_cases(extended, high_order):
        for order, cutoff, body in specifications:
            print(f"primitive {name} FC{order}", flush=True)
            result = validate_primitive_case(
                primitive,
                order=order,
                cutoff=cutoff,
                max_body_order=body,
                symprec=symprec,
            )
            primitive_results.append({"case": name, **result})

    finite_results = []
    primitive = bulk("NaCl", "rocksalt", a=5.64)
    sizes = (2, 3, 4) if extended else (2, 3)
    for size in sizes:
        print(f"finite pair NaCl {size}x{size}x{size}", flush=True)
        reference = build_supercell(primitive, (size, size, size))
        frame = ReferenceFrame.from_atoms(primitive, reference, symprec=1e-5)
        finite_results.append(
            {"case": f"NaCl-{size}x{size}x{size}", **validate_finite_pair_case(frame)}
        )

    reference = build_supercell(primitive, (2, 2, 2))
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: 3.0},
        periodic_fc2_completion=True,
    )
    completion = validate_periodic_completion(fitter.calculations[0])
    return {
        "status": "pass",
        "extended": extended,
        "high_order": high_order,
        "primitive": primitive_results,
        "finite_pair": finite_results,
        "periodic_completion": completion,
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--high-order", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(extended=arguments.extended, high_order=arguments.high_order),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
