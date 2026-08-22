# MLFCS

MLFCS is an ASE-first Python library for symmetry-reduced force constants from atomic forces.
It supports finite differences and force-only fitting from FC2 through arbitrary supported order,
with sparse native storage and explicit output views.

## Choose a workflow

| Goal | Start here |
|---|---|
| Calculate FC2/FC3/FC4 from an ASE calculator | [Finite differences](workflows/finite-difference.md) |
| Use VASP or another external force program | [External calculators](workflows/external-calculators.md) |
| Fit force constants from displaced or MD snapshots | [Force-only fitting](workflows/fitting.md) |
| Generate temperature-dependent FC2 by sampling | [SSCHA](workflows/sscha.md) |
| Apply a quartic loop correction | [Loop-SCPH](workflows/scph.md) |
| Export to another phonon/transport program | [Formats](formats/index.md) |

## Three rules before starting

1. Decide which downstream program will read the result.
2. Use that program's primitive and reference supercell when possible.
3. Keep the reference atom order unchanged from structure generation through force collection.

MLFCS validates equivalent representations at export, but it does not silently redefine the
primitive cell, enlarge a supercell, or rotate a structure.

## Installation

```bash
uv sync
```

The base package does not install phonopy, phono3py, ShengBTE, ALAMODE, or a calculator.
Install those tools only for the workflow that consumes the exported force constants.

See [Getting started](getting-started/index.md) for the smallest complete example and
[Cases](cases/index.md) for reproducible material workflows.
