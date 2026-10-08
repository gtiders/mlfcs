"""Run serial ASR prototype checks and report measured memory and factor sizes."""

import argparse
import json
import resource
from time import perf_counter

import numpy as np
from ase.build import bulk
from ase.io import read

from mlfcs import ClusterSpace
from mlfcs.cluster_space.acoustic import lattice_acoustic_equations, prepare_coordinates


def run(kind, initialize_asr=False):
    """Validate a Si or Ba model without fitting or transport calculations."""
    if kind == "ba":
        atoms = read("docs/notebooks/data/ba8ga16ge30/primitive.vasp")
        cutoffs = {2: 6.0, 3: 6.0, 4: 3.0}
    else:
        atoms = bulk("Si", "diamond", a=5.43)
        cutoffs = {2: 3.0, 3: 3.0, 4: 3.0}
        if kind == "sheared":
            atoms.set_cell(
                np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ atoms.cell.array, scale_atoms=False
            )
            atoms.wrap()
    started = perf_counter()
    space = ClusterSpace(
        atoms, cutoffs=cutoffs, max_body_orders={2: 2, 3: 3, 4: 4}, asr=initialize_asr
    )
    print(
        json.dumps(
            {
                "model": kind,
                "stage": "initialization",
                "asr": initialize_asr,
                "seconds": perf_counter() - started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        ),
        flush=True,
    )
    for order in (2, 3, 4):
        started = perf_counter()
        equations, maps = lattice_acoustic_equations(space, order)
        stats = {
            "model": kind,
            "order": order,
            "shape": equations.shape,
            "nnz": equations.nnz,
            "csr_bytes": equations.data.nbytes + equations.indices.nbytes + equations.indptr.nbytes,
            "assembly_seconds": perf_counter() - started,
        }
        print(json.dumps(dict(stats, stage="assembled")), flush=True)
        started = perf_counter()
        factor = (
            space.acoustic_coordinates(order)
            if initialize_asr
            else prepare_coordinates(equations, maps)
        )
        stats.update(
            rank=len(factor.pivots),
            nullity=len(factor.free),
            upper_nnz=len(factor.values),
            retained_bytes=sum(
                a.nbytes
                for a in {
                    id(a): a
                    for a in (
                        factor.indptr,
                        factor.indices,
                        factor.values,
                        factor.pivots,
                        factor.free,
                        *factor.physical_maps,
                    )
                }.values()
            ),
            physical_maps=len(factor.physical_maps),
            unique_physical_maps=len({id(a) for a in factor.physical_maps}),
            largest_factor_coefficient=int(np.max(np.abs(factor.values), initial=0)),
            factor_seconds=perf_counter() - started,
        )
        del equations, maps
        stats.update(peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(json.dumps(dict(stats, stage="certified")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("si", "sheared", "ba"))
    parser.add_argument("--initialize-asr", action="store_true")
    args = parser.parse_args()
    run(args.model, args.initialize_asr)
