"""Optional convenience tools that build structures for the core APIs.

This package is a leaf: it may import :mod:`mlfcs.structure`, but no core package may import it.
The core APIs take explicit structures and never construct or reorder them implicitly. Callers
import the supercell construction and external-alignment tools from :mod:`mlfcs.tools.supercell`.
"""

from __future__ import annotations

__all__: list[str] = []
