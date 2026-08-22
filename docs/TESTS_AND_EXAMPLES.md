# Tests and examples policy

[中文](TESTS_AND_EXAMPLES_ZH.md) | English

## Target layout

MLFCS will keep two validation entry points:

| Location | Responsibility | Ordinary CI |
|---|---|---|
| `tests/` | Deterministic mathematics, APIs, invariants, and minimal format contracts | Required |
| `examples/` | Public API examples and real-material scientific cases | Lightweight checks only |

The current `assets/` tree will be merged into `examples/cases/`. External-program inputs,
results, and training data may remain as case references, but no longer act as strict pytest
oracles. Cases do not require `case.json`, SHA-256 manifests, or hash gates. One complete README is
the only mandatory case description.

## Boundary of `tests/`

Keep unit tests for supercell relations, periodic geometry, orbits, finite differences, Wick
conversion, constraints, sparse IFCs, and writer semantics. Integration tests exercise minimal
public workflows: `sow -> reap`, fitting, HDF5 roundtrips, SSCHA, and explicit target export.
Prefer hand-checkable models, independently differentiated energies, convergence order, sum rules,
and representation invariants. An optional official-reader test may establish only readability,
shape, and labels.

Do not add complete phonopy, phono3py, ALAMODE, hiphive, or ShengBTE material results as truth
arrays. Tests do not contain large training sets, transport directories, potential training, full
material fits, or old MLFCS outputs as oracles. A defect found by an external case becomes a small
independent regression test; the whole material workflow does not run in ordinary CI.

Tests use fixed random seeds, minimal structures, and tolerances justified by units or numerical
analysis. They do not read `examples/cases/`, depend on execution order, or use developer paths.

## Two kinds of examples

Top-level `examples/*.py` files demonstrate one small public-API task and remain material-neutral.
They use no private MLFCS modules or absolute paths. Optional dependencies are documented with
`uv run --with ...`. Cases call these shared tools by argument instead of copying scripts.

Real-material cases use:

```text
examples/cases/<Material>/<case>/
  README.md
  structures/{primitive.vasp,reference.vasp}
  workflow/{run_mlfcs.py,compare.py}
  fitting/train.extxyz
  finite_difference/
    POSCAR-unitcell
    mlfcs-plan.json
    structures/POSCAR-001 ...
    calculations/POSCAR-001/vasprun.xml ...
    forces.npz
  results/{mlfcs,reference/<software>}/
  observables/{phonons,thermal_conductivity}/
```

Only directories needed by a case are created. Material names use chemical capitalization; case
names describe physics, such as `harmonic-fit`, `fc3-fd`, or `shengbte-kappa`.

## README-only case description

No uniform manifest or checksum file is required. Each README records the scientific question and
limits; primitive, reference, supercell, and atom order; orders, cutoffs, body orders,
displacements, constraints, and units; MLFCS commands; third-party source, version, and parameters;
artifact roles; downstream NAC, q mesh, broadening, isotope, boundary, and iterative/RTA settings;
and known differences or unconverged quantities.

URLs, revisions, and hashes may be recorded when useful, but are not mandatory gates. Third-party
results retain their native formats. MLFCS results preferentially retain native HDF5 v2, plus only
the downstream formats needed by the case.

## Fitting datasets

The MLFCS fitting input is strict ASE-readable extxyz:

```text
fitting/train.extxyz
```

Every frame has the same atom count, species labels, atom order, and lattice as
`structures/reference.vasp`, and carries `(N, 3)` forces in ASE calculator results. MLFCS does not
silently reorder or recenter snapshots. Energy, stress, temperature, grouping, and provenance may
be extxyz metadata. Separate sources may use names such as `train-aimd.extxyz` and
`train-random.extxyz`, with their combination documented in the README.

Original third-party training formats may be retained as reference/source material, but the actual
MLFCS workflow consumes extxyz so cases do not maintain bespoke parsers.

## Finite-difference datasets

Finite-difference data are not converted to extxyz. They retain the ordered sow workspace:

```text
finite_difference/
  POSCAR-unitcell
  mlfcs-plan.json
  structures/POSCAR-001
  structures/POSCAR-002
  calculations/POSCAR-001/vasprun.xml
  calculations/POSCAR-002/vasprun.xml
  forces.npz
```

Structures retain reference atom order, and calculation directories use matching names. Natural
filename order is for inspection; the authoritative order is the `filenames` list in
`mlfcs-plan.json` and continuous `configuration_ids` in `forces.npz`. Jobs may finish out of order,
but collection restores sow order. No external program may reorder atoms between POSCAR and force.

When raw outputs are too large or cannot be redistributed, retain structures, the plan, collected
forces, required input descriptions, and final native HDF5. The README states where raw results are
archived. Proprietary POTCARs, caches, electronic iteration logs, and irrelevant intermediates do
not enter cases.

## Scientific evidence

Native third-party results may remain in examples as reference evidence, not automatic truth.
Comparisons document primitive and supercell conventions, atom order, units, images, cutoffs, and
constraints. Confidence combines force errors, invariants, aligned IFC or dynamical-matrix
comparisons, phonons and NAC, transport convergence, and where useful experiment or literature.

Phonon cases retain machine-readable q points and frequencies, not only PNG. Conductivity cases
retain temperature-dependent tensors and convergence data, not one number. Different IFCs with
converged equivalent observables are reported as a method difference. Similar final observables
without aligned upstream conventions do not prove a writer correct.

## Migration plan awaiting approval

No existing data move or deletion occurs before approval:

1. Create `examples/cases/<Material>/<case>/`.
2. Split `assets/SI` into Si harmonic, FC3 finite-difference, and ShengBTE conductivity cases.
3. Move ALAMODE Si and SrTiO3 into fitting cases with extxyz as the MLFCS input.
4. Correct K3Au3Sb2 naming according to its actual composition and task.
5. Deduplicate plotting scripts and call `examples/plot_phonon_band.py` from cases.
6. Write each README and retain necessary external results plus MLFCS native HDF5.
7. Produce an explicit deletion list before removing caches, duplicates, proprietary files, or
   irrelevant intermediates.
8. Move material comparisons out of `tests/reference` only after adding minimal synthetic
   regressions for defects they exposed.
9. Remove the reference marker, CI job, and dependencies after migration.
10. Run unit/integration tests and manually review migrated phonon and conductivity evidence.

Until this plan is approved, `assets/`, `tests/reference/`, and their results remain in place.
