#!/usr/bin/env python3
"""Benchmark continuous-order ForceConstantFitter initialization."""
from __future__ import annotations
import argparse, json, resource, time
from io import StringIO
from pathlib import Path
from ase.io import read
from mlfcs import ForceConstantFitter, build_supercell
from mlfcs.interactions.primitive.candidates import resolve_primitive_cutoff
ROOT = Path(__file__).resolve().parents[2]
SHELL_CUTOFFS = {2: -8, 3: -6, 4: -3, 5: -2, 6: -2}
CASES = {"POSCAR-Ag2Bi6S10": {"primitive": ROOT / "POSCAR", "reference_matrix": (3, 3, 3)}}
def read_structure(path: Path):
    return read(StringIO(path.read_text(encoding="utf-8").lstrip("\n")), format="vasp")
def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
def benchmark_case(name: str, maximum_order: int) -> dict[str, object]:
    case = CASES[name]; primitive = read_structure(case["primitive"]); reference = build_supercell(primitive, case["reference_matrix"], symprec=1e-4)
    orders = tuple(range(2, maximum_order + 1)); cutoffs = {o: resolve_primitive_cutoff(primitive, SHELL_CUTOFFS[o], reference=reference) for o in orders}; bodies = {o: (3 if o >= 5 else None) for o in orders}
    started = time.perf_counter(); fitter = ForceConstantFitter(primitive, reference, orders=orders, cutoffs=cutoffs, max_body_orders=bodies, symprec=1e-4); elapsed = time.perf_counter() - started
    result = {"case": name, "orders": list(orders), "requested_cutoff_shells": {str(o): SHELL_CUTOFFS[o] for o in orders}, "resolved_cutoffs_angstrom": {str(o): cutoffs[o] for o in orders}, "max_body_orders": {str(o): bodies[o] for o in orders}, "primitive_atoms": len(primitive), "reference_atoms": len(reference), "parameters": fitter.n_parameters, "initialization_seconds": elapsed, "peak_rss_mib": peak_rss_mib()}
    print(json.dumps(result, sort_keys=True), flush=True); return result
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--case", choices=tuple(CASES), required=True); parser.add_argument("--order", choices=(3,4,5,6), type=int, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); args.output.write_text(json.dumps(benchmark_case(args.case, args.order), indent=2, sort_keys=True) + "\n", encoding="utf-8")
if __name__ == "__main__": main()
