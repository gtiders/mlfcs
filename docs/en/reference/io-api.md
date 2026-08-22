---
title: I/O API
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# I/O API

Document native HDF5 reading and the explicit multi-format writer with target requirements.

~~~python
read_hdf5(source: str | Path) -> ForceConstants

write_force_constants(
    force_constants: ForceConstants,
    target: str | Path,
    *,
    format: str,
    order: int | None = None,
    primitive: Atoms | None = None,
    supercell: Atoms | None = None,
) -> None
~~~

`format` is always explicit. Native HDF5 needs no target; external dense or format-limited outputs may require a validated primitive or supercell view.
