#!/usr/bin/env python3
"""Benchmark hiPhive ClusterSpace initialization for POSCAR material."""
from __future__ import annotations
import argparse, json, resource, time
from io import StringIO
from pathlib import Path
from ase.io import read
from hiphive import ClusterSpace
from hiphive.cutoffs import CutoffMaximumBody
from mlfcs import build_supercell
from mlfcs.interactions.primitive.candidates import resolve_primitive_cutoff
ROOT = Path(__file__).resolve().parents[2]
SHELL_CUTOFFS = {2: -8, 3: -6, 4: -3, 5: -2, 6: -2}
def read_structure(path: Path):
    return read(StringIO(path.read_text(encoding="utf-8").lstrip("\n")), format="vasp")
def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
def benchmark(maximum_order: int) -> dict[str, object]:
    primitive = read_structure(ROOT / "POSCAR"); reference = build_supercell(primitive, (3, 3, 3), symprec=1e-4)
    orders = tuple(range(2, maximum_order + 1)); cutoffs = [resolve_primitive_cutoff(primitive, SHELL_CUTOFFS[o], reference=reference) for o in orders]
    started = time.perf_counter(); clusters = CutoffMaximumBody(cutoffs, 3 if maximum_order >= 5 else maximum_order); space = ClusterSpace(primitive, clusters, symprec=1e-4); elapsed = time.perf_counter() - started
    result = {"case": "POSCAR-Ag2Bi6S10", "orders": list(orders), "shell_cutoffs": {str(o): SHELL_CUTOFFS[o] for o in orders}, "cutoffs_angstrom": cutoffs, "max_body_order": 3 if maximum_order >= 5 else maximum_order, "primitive_atoms": len(primitive), "clusters": len(space.cluster_list), "parameters": space.n_dofs, "initialization_seconds": elapsed, "peak_rss_mib": peak_rss_mib()}
    print(json.dumps(result, sort_keys=True), flush=True); return result
def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--order", choices=(3,4,5,6), type=int, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); args.output.write_text(json.dumps(benchmark(args.order), indent=2, sort_keys=True) + "\n", encoding="utf-8")
if __name__ == "__main__": main()
