---
title: I/O API
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# Read and write API

## `read_hdf5`

```python
read_hdf5(source: str | Path) -> ForceConstants
```

Reads native MLFCS HDF5 schema v4. The file stores the primitive structure, `symprec`, metadata,
and sparse exact integer-labelled sites, translations, and tensors. It is not the dense HDF5
format used by phonopy or phono3py. Older native schemas are rejected explicitly.

## `write_force_constants`

```python
write_force_constants(
    force_constants: ForceConstants,
    target: str | Path,
    *,
    format: str,
    order: int | None = None,
    primitive: Atoms | None = None,
    supercell: Atoms | None = None,
) -> None
```

`format` is mandatory; the filename extension never selects a format. Native `hdf5` writes every
sparse order using schema v4. `phonopy` and `phonopy_hdf5` write dense FC2, `phono3py_hdf5` writes
dense FC3, and the ShengBTE and ALAMODE writers use their documented order-specific formats.

Target realization always uses an explicit primitive/reference relation. A writer never constructs
a supercell from a matrix or guesses one from an interaction cutoff.
