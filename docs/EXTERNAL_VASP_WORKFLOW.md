# External VASP sow/collect/reap workflow

[中文](EXTERNAL_VASP_WORKFLOW_ZH.md)

MLFCS does not launch VASP or prescribe INCAR, KPOINTS, POTCAR, scheduler, or convergence
settings. It defines the displaced structures and the exact positional contract by which their
forces return. The complete example is [`examples/vasp_external_fc3.py`](../examples/vasp_external_fc3.py).

At the API level, an exact positional workflow is sufficient: force set `i` must correspond to
structure `i` returned by `sow()`, and `reap(forces)` needs neither IDs nor a plan
hash. This example deliberately adds a manifest and hash as an operational safety layer for batch
jobs, restarts, missing results, and long-term provenance.

## 1. Sow ordered structures

Start from a primitive VASP structure and choose the same physical parameters that will later be
used for reconstruction:

```bash
uv run python examples/vasp_external_fc3.py sow POSCAR fc3-work \
  --supercell 3 3 3 \
  --cutoff -6 \
  --displacement 0.01
```

This creates:

```text
fc3-work/
├── POSCAR-unitcell
├── mlfcs-plan.json
└── structures/
    ├── POSCAR-001
    ├── POSCAR-002
    └── ...
```

`POSCAR-001` is configuration ID 0, `POSCAR-002` is ID 1, and so forth. The manifest stores the
supercell, cutoff, displacement, reference atom order, configuration count, and filename sequence.
Do not rename, omit, deduplicate, or reorder configurations after sowing.

## 2. Prepare and submit VASP calculations

Create one calculation directory per generated filename. A typical preparation loop is:

```bash
mkdir -p fc3-work/calculations
for structure in fc3-work/structures/POSCAR-*; do
  name=$(basename "$structure")
  directory="fc3-work/calculations/$name"
  mkdir -p "$directory"
  cp "$structure" "$directory/POSCAR"
  cp INCAR KPOINTS POTCAR "$directory/"
done
```

The expected completed layout is:

```text
fc3-work/calculations/
├── POSCAR-001/vasprun.xml
├── POSCAR-002/vasprun.xml
└── ...
```

Submit these directories using the local scheduler. Every run must be a consistent static force
calculation: use identical electronic settings, sufficiently tight force convergence, and no ionic
relaxation. MLFCS intentionally does not provide the submission script because scheduler and VASP
configuration are site-specific.

## 3. Collect forces

After every calculation has a complete `vasprun.xml`, collect forces in manifest order:

```bash
uv run python examples/vasp_external_fc3.py collect \
  fc3-work fc3-work/calculations
```

The command fails if any result is missing. It reads the final ASE force array from every
`vasprun.xml` and writes `fc3-work/forces.npz` with configuration IDs and atom order.
The original VASP directories remain the authoritative raw results.

## 4. Reap and export

Reconstruct FC3 and write the default faithful ShengBTE representation:

```bash
uv run python examples/vasp_external_fc3.py reap \
  fc3-work FORCE_CONSTANTS_3RD \
  --format shengbte
```

Strict translational ASR is enabled by default. Pass `--no-asr` to retain the raw finite-
difference result. ShengBTE serializes the reconstructed sparse physical support directly;
there is no legacy thirdorder compatibility mode. Alternative FC3 outputs include generic sparse
`hdf5`, `numpy`, and full `phono3py_hdf5`.

## Recovery and audit rules

- Keep `mlfcs-plan.json`, `POSCAR-unitcell`, and `forces.npz` together.
- Archive the original VASP inputs and `vasprun.xml` files separately.
- If the structure, supercell, cutoff, displacement, implementation, or symmetry tolerance changes,
  sow again instead of mixing datasets.
- If a scheduler returns jobs out of order, directory names restore the exact sequence during
  collection.
- Forces must correspond to the reference atom order written in each generated POSCAR. Never apply
  a separate atom sort between VASP and collection.
