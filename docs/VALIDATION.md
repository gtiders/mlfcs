# Numerical validation and continuous integration

English | [中文](VALIDATION_ZH.md)

## Validation goals

The suite independently checks the finite-difference reconstruction, atom and translation
conventions, and physical equivalence of exported representations. Merely producing a readable
file is not treated as numerical validation.

## AlN FC3 reference

The AlN reference originates from phono3py's `example/AlN-rd` dataset. A pypolymlp 0.20.4
potential was fitted once to all 200 structures and then used to produce forces for both MLFCS
and phono3py. Both paths use a 2x2x2 supercell, 0.01 Angstrom displacements, and complete
minimum-image pair coverage. MLFCS uses a radius of `5.8760168278` Angstrom; phono3py uses its
untruncated default.

The raw comparison disables MLFCS ASR so that constraint projections do not obscure the
finite-difference comparison. hiphive is used only as an independent representation adapter.
On 32000 atom triplets, the maximum difference is approximately `0.01692 eV/Angstrom^3`, the RMS
difference `0.000464 eV/Angstrom^3`, the relative L2 error `2.87e-4`, and the correlation
`0.9999999615`.

## AlN FC2 and ASR references

The same potential compares MLFCS FC2 with phonopy's traditional solver. MLFCS uses 12 central
differences while phonopy uses four symmetry-selected structures, so the final tensors—not the
displacement plans—are compared. Without MLFCS ASR, the maximum difference is about
`0.003326 eV/Angstrom^2`, RMS `0.000357 eV/Angstrom^2`, and relative L2 error `1.48e-4`.

A separate FC3 test compares strict MLFCS ASR with phono3py
`symmetrize_fc3(level=3)`. Their residuals fall to approximately `2.58e-13` and
`5.53e-14 eV/Angstrom^3`, respectively. The projected tensors have relative L2 difference
`0.0527%` and correlation `0.9999998627` on their shared support.

## CI layers

- `unit-and-api`: Ruff, formatting, and all non-reference tests on Python 3.12 and 3.13;
- `scientific-reference`: provenance checks and independent FC2/FC3 comparisons, run serially;
- `package`: source-distribution and wheel builds.

BLAS, OpenMP, and the JAX CPU backend are restricted to one thread in CI to keep memory use
predictable. Potential fitting is a maintainer regeneration task, not part of ordinary CI.
See [`tests/README.md`](../tests/README.md) for commands and fixture organization.

## Provenance and regeneration

Pinned upstream commits, licenses, hashes, versions, and regeneration commands are stored beside
each fixture under [`tests/reference`](../tests/reference/). The scripts in
[`reference_tools`](../reference_tools/README.md) regenerate those fixtures; MLFCS does not depend
on hiphive, phonopy, phono3py, pypolymlp, or symfc at base-package runtime.
