---
title: Complete SNF mapping research
audience:
  - developer
status: research
code_verified: 4.0.0a4
---

# Can complete SNF unify finite-supercell mapping?

Complete Smith normal form can express the translation quotient as finite cyclic coordinates and can clarify direct/reciprocal group structure. In the current $3\times3$ workload it adds transformation bookkeeping without replacing the need for real-space representatives, while the available Python interfaces do not provide a sufficiently stable complete decomposition for the production mapping chain.

## Decision

HNF remains the sole production quotient reduction and representative system. SNF remains a research diagnostic and is not written into IFC keys, HDF5, q-point APIs, or atom mapping.
