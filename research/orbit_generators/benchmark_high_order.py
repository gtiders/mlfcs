#!/usr/bin/env python3
"""Benchmark FC5/FC6 key enumeration without tensor bases or fitting."""

from __future__ import annotations

import argparse
import resource
from itertools import permutations
from math import factorial
from time import perf_counter

from ase.build import bulk
from prototype import _action_generators, _transform_key

from mlfcs.interactions.enumerate import (
    _canonical_key,
    _compatible_tails,
    _primitive_neighbors,
    resolve_primitive_cutoff,
)
from mlfcs.interactions.keys import InteractionKey
from mlfcs.structure.symmetry import PrimitiveSymmetryOperations


def _memory_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _representatives(order: int, cutoff: float, maximum_body_order: int):
    primitive = bulk("Si", "diamond", a=5.43)
    symmetry = PrimitiveSymmetryOperations.from_atoms(primitive, symprec=1e-5)
    radius = resolve_primitive_cutoff(primitive, cutoff)
    neighbors = _primitive_neighbors(primitive, radius)
    seen: set[InteractionKey] = set()
    candidate_count = 0
    started = perf_counter()
    for anchor in range(len(primitive)):
        for tail in _compatible_tails(neighbors[anchor], order - 1, primitive, radius):
            key = InteractionKey.from_labels(((anchor, 0, 0, 0), *tail))
            if len(set(key.labels)) > maximum_body_order:
                continue
            candidate_count += 1
            seen.add(_canonical_key(key, symmetry))
    return primitive, symmetry, tuple(sorted(seen)), candidate_count, perf_counter() - started


def _exhaustive_images(representative, symmetry, axis_permutations):
    images = set()
    for operation in range(symmetry.size):
        transformed = tuple(
            symmetry.transform_label(operation, label) for label in representative.labels
        )
        for permutation in axis_permutations:
            images.add(InteractionKey.from_labels(transformed[axis] for axis in permutation))
    return images


def _generator_images(representative, symmetry, generators):
    reached = {representative}
    pending = [representative]
    while pending:
        key = pending.pop()
        for generator in generators:
            image = _transform_key(key, symmetry, generator)
            if image not in reached:
                reached.add(image)
                pending.append(image)
    return reached


def benchmark(order: int, cutoff: float, maximum_body_order: int) -> None:
    _primitive, symmetry, representatives, candidates, cluster_seconds = _representatives(
        order, cutoff, maximum_body_order
    )
    print(
        f"Si FC{order}: cutoff={cutoff:g} A, max_body={maximum_body_order}, "
        f"candidates={candidates}, representatives={len(representatives)}, "
        f"cluster_seconds={cluster_seconds:.6f}, peak_rss_mib={_memory_mib():.1f}",
        flush=True,
    )

    axis_permutations = tuple(permutations(range(order)))
    exhaustive_images = 0
    started = perf_counter()
    for index, representative in enumerate(representatives, start=1):
        exhaustive_images += len(_exhaustive_images(representative, symmetry, axis_permutations))
        if index % 100 == 0:
            print(f"  exhaustive representatives: {index}/{len(representatives)}", flush=True)
    exhaustive_seconds = perf_counter() - started

    generators = _action_generators(symmetry, order)
    generator_images = 0
    generator_edges = 0
    started = perf_counter()
    for index, representative in enumerate(representatives, start=1):
        images = _generator_images(representative, symmetry, generators)
        generator_images += len(images)
        generator_edges += len(images) * len(generators)
        if index % 100 == 0:
            print(f"  generator representatives: {index}/{len(representatives)}", flush=True)
    generator_seconds = perf_counter() - started

    if generator_images != exhaustive_images:
        raise AssertionError(
            f"image count mismatch: exhaustive={exhaustive_images}, generator={generator_images}"
        )
    print(
        f"  images={exhaustive_images}, symmetry_operations={symmetry.size}, "
        f"permutations={factorial(order)}, exhaustive_actions="
        f"{len(representatives) * symmetry.size * factorial(order)}",
        flush=True,
    )
    print(
        f"  exhaustive_seconds={exhaustive_seconds:.6f}, "
        f"generator_edges={generator_edges}, "
        f"generator_seconds={generator_seconds:.6f}, "
        f"generator_over_exhaustive={generator_seconds / exhaustive_seconds:.3f}, "
        f"peak_rss_mib={_memory_mib():.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("order", type=int, choices=(5, 6))
    parser.add_argument("--cutoff", type=float, default=4.6)
    parser.add_argument("--maximum-body-order", type=int, default=3)
    arguments = parser.parse_args()
    benchmark(arguments.order, arguments.cutoff, arguments.maximum_body_order)


if __name__ == "__main__":
    main()
