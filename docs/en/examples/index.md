---
title: Examples
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Examples

Examples are reproducible physical and numerical evidence, not substitutes for tutorials. Each repository case owns its inputs, scripts, provenance, expected outputs, and retained figures; generated caches and large force-constant files remain local unless explicitly identified as reference data.

## Finite differences and fitting

- [Si finite differences](si-finite-difference.md), [fitting](si-fitting.md), and [transport handoff](si-transport.md)
- [K4As4Pt2 finite differences](k4as4pt2-finite-difference.md), [fitting](k4as4pt2-fitting.md), and [transport](k4as4pt2-transport.md)
- [SnSe high-order fitting](snse-fitting.md)
- [Ba8Ga16Ge30 temperature-dependent fitting](ba8ga16ge30-md-fitting.md) and [transport](ba8ga16ge30-transport.md)

## Constraints and temperature-dependent phonons

- [MoS2](mos2-rotational.md) and [graphene](graphene-rotational.md) rotational constraints
- [K4As4Pt2 SCPH](k4as4pt2-scph.md) and [SSCHA](k4as4pt2-sscha.md)
- [KCl SSCHA](kcl-sscha.md)

## Mapping regression

- [Non-diagonal supercell regression](non-diagonal-supercell.md)

Run case scripts from the repository root with `uv run`. Read each case README before execution because optional calculators and downstream applications are intentionally not base dependencies.
