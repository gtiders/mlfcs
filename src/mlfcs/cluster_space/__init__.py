"""Primitive-only cluster spaces.

The package surface is ``ClusterSpace`` alone; the immutable records that make
it up live in :mod:`mlfcs.cluster_space.records`, and the assembly, search and
exact-arithmetic stages in their own modules.
"""

from mlfcs.cluster_space.records import ClusterSpace

__all__ = ["ClusterSpace"]
