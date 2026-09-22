"""Optional convenience tools that build structures for the core APIs.

This package is a leaf: it may import :mod:`mlfcs.structure`, but no core package may import it.
The core APIs take an explicit reference supercell and never build one themselves, so a caller
that wants the phonopy old-style ordering imports it from :mod:`mlfcs.tools.supercell` by name.
"""

from __future__ import annotations

__all__: list[str] = []
