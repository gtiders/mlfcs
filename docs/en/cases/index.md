# Examples

English | [中文]

The material cases are reproducible workflow fixtures. Their README files are the source of
commands, input provenance, expected outputs, and third-party reference data.

## Material cases

- [Si](https://github.com/gtiders/mlfcs/tree/main/examples/cases/Si/README.md): finite-difference and force-only fitting workflows.
- [K4As4Pt2](https://github.com/gtiders/mlfcs/tree/main/examples/cases/K4As4Pt2/fitting/README.md): FC2--FC4 fitting and loop-SCPH.
- [Ba8Ga16Ge30](https://github.com/gtiders/mlfcs/tree/main/examples/cases/Ba8Ga16Ge30/fitting/README.md): public hiPhive training data.
- [KCl](https://github.com/gtiders/mlfcs/tree/main/examples/cases/KCl/sscha/README.md): native SSCHA reference.
- [MoS2 and graphene](https://github.com/gtiders/mlfcs/tree/main/examples/cases/rotational_sum_rules/MoS2_monolayer/README.md):
  second-order rotational constraints.

These scripts demonstrate public MLFCS APIs. Run them from the repository root with `uv run`.
They are examples, not an MLFCS command-line interface.
New scripts and material cases follow the [tests and examples policy].

## Direct calculators

- [`basic_fc2.py`] relaxes and evaluates a small system with ASE EMT, then writes
  FC2.
- [`nep89_orders.py`] loads a user-supplied NEP89 model through calorine and
  computes one or more orders. The model remains an external user dependency.

For example:

```bash
uv run python examples/nep89_orders.py POSCAR nep89.txt \
  --orders 2 3 --supercell 2 2 2 --cutoff -3 --output-directory results
```

## External VASP calculations

[`vasp_external_fc3.py`] is a complete three-stage reference:

```bash
uv run python examples/vasp_external_fc3.py sow POSCAR fc3-work \
  --supercell 3 3 3 --cutoff -6

# Create calculations/POSCAR-001, calculations/POSCAR-002, ...;
# copy each matching POSCAR, run VASP, and retain vasprun.xml.

uv run python examples/vasp_external_fc3.py collect \
  fc3-work fc3-work/calculations
uv run python examples/vasp_external_fc3.py reap \
  fc3-work FORCE_CONSTANTS_3RD --format shengbte
```

The helper never submits VASP. Site-specific INCAR, KPOINTS, POTCAR, and scheduler setup remain
the user's responsibility. See the [complete workflow guide]
before using externally calculated forces.
