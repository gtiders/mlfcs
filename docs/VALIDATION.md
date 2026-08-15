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

When an ALAMODE `anphon` executable is installed, the optional integration
test below validates that its real harmonic reader accepts both a minimal
MLFCS-produced FCSXML document and the two-way 27-image expansion at a
half-supercell boundary. Set `MLFCS_ANPHON` when it is not on `PATH`:

```bash
MLFCS_ANPHON=/path/to/anphon uv run pytest -q tests/integration/test_alamode_anphon.py
```
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

## Public hiPhive BaGaGe complex-material benchmark

The optional complex-material benchmark uses hiPhive's public 200-snapshot
Ba8Ga16Ge30 clathrate dataset: 100 Monte-Carlo-rattled snapshots plus 50 MD-based
snapshots at each of 300 and 650 K. Both implementations use the published
two-body FC2+FC3+FC4 model, cutoffs `[5.40, 4.35, 4.35]` Angstrom, the R3
54-atom cell, and translational ASR. The common physical space has 25,495
coefficients and the exact ASR null-space has 6,052 fitting coordinates.

On all 200 structures (training fit, rather than the publication's 10-fold
cross-validation), hiPhive gives a `49.10 meV/Angstrom` force RMSE and MLFCS
gives `57.60 meV/Angstrom`. The difference is expected: hiPhive fits ordinary
Taylor features, whereas MLFCS fits covariance-orthogonal Wick features and
then converts the result to Taylor IFCs. After atom and tensor-axis alignment,
the FC2/FC3/FC4 relative RMS differences are `1.62%`, `18.63%`, and `47.64%`.
This is therefore a demanding *method-comparison* result, not an assertion of
byte- or tensor-level identity. All clusters matched during comparison.

The same 8,192 sampled-feature diagnostic shows why the orthogonal basis is
useful but not magical: Taylor versus Wick linear/cubic correlations have mean
`0.06409` versus `0.06280`, RMS `0.08094` versus `0.07867`, and maximum
`0.68239` versus `0.40341`.

A stricter 2026-08-15 check streams the *complete* ASR-reduced physical design
for the same FC2+FC3+FC4 model and compares only its FC2--FC4 cross Gram block.
It therefore includes the actual symmetry basis, two-body support, ASR null
space, and column normalization used by the solver. Wick reduces the maximum
pairwise normalized FC2--FC4 correlation from `0.51662` to `0.21352` and its
RMS from `0.01683` to `0.00768`; the column-normalized joint Gram condition
number falls from `2.63e6` to `1.37e6`. However, the maximum *subspace* canonical
correlation changes from `0.94515` to `0.96235`. Thus Wick demonstrably reduces
direct column-level coupling in this case, but does not guarantee that the two
whole constrained subspaces become more orthogonal.

Resource measurements were made serially with `/usr/bin/time -v` on the
development CPU host. MLFCS first builds a 279.4 MiB reduced Gram matrix after
pre-parameterizing ASR; the successful cache-recovery solve took 64.78 s and
peaked at 1.46 GiB RSS. Its preceding cold Gram build took 65.96 s and reached
2.01 GiB before the intentionally too-small 1,000-iteration solver limit was
raised to 10,000. The completed hiPhive baseline took approximately 13 minutes
and its observed peak RSS was 6.48 GiB while materializing its explicit design
matrix. These are host-specific measurements, not universal performance claims.

The Gram recovery cache was measured independently in an empty temporary
directory: a cold 200-snapshot BaGaGe Gram took `67.99 s`, whereas an immediate
verified warm hit took `0.0689 s` (about `987x`). The end-to-end command still
took 115.70 s because symmetry/ASR parameterization and JAX-program preparation
are intentionally recomputed; the cached Gram itself is not slower in a
meaningful sense than the earlier 65.96 s cold measurement (about 3% variation).

```bash
/usr/bin/time -v uv run python reference_tools/benchmark_hiphive_examples.py bagage-hiphive --validation-split 0.0
/usr/bin/time -v uv run python reference_tools/benchmark_hiphive_examples.py bagage-mlfcs --validation-split 0.0
uv run python reference_tools/benchmark_hiphive_examples.py bagage-compare
uv run python reference_tools/benchmark_hiphive_examples.py bagage-wick
uv run python reference_tools/benchmark_hiphive_examples.py bagage-collinearity
/usr/bin/time -v uv run python reference_tools/benchmark_hiphive_examples.py bagage-gram-cache
```

## Native SSCHA references

Analytic harmonic models independently check classical covariance, quantum zero-point variance,
imaginary-mode handling, optional displacement clipping, and native FC2 recovery from sampled
forces. A development-only phonopy oracle additionally compares commensurate-q frequencies and
the sampled quantum covariance. Phonopy is used only by this reference test, not by the SSCHA
implementation or the base runtime environment.

An end-to-end KCl reference additionally uses phonopy's own 120-structure pypolymlp potential,
eight-atom conventional cell, 2x2x2 supercell, 300 K temperature, and seed 42. The serial CI
variant uses ten snapshots and one canonical iteration: its K self block is about
`2.1625 eV/Angstrom^2`, inside phonopy's official `2.1 +/- 0.1` acceptance range. Its normalized
free energy is about `-0.0949 eV` per primitive cell, within `3.7 meV` of phonopy's dense-mesh
three-iteration reference. The remaining difference includes sampling noise and the deliberate
commensurate-q versus dense-mesh harmonic-free-energy convention.

## CI layers

- `unit-and-api`: Ruff, formatting, and all non-reference tests on Python 3.12 and 3.13;
- `scientific-reference`: provenance checks and independent FC2/FC3 and harmonic-sampling
  comparisons, run serially;
- `package`: source-distribution and wheel builds.

BLAS, OpenMP, and the JAX CPU backend are restricted to one thread in CI to keep memory use
predictable. Potential fitting is a maintainer regeneration task, not part of ordinary CI.
See [`tests/README.md`](../tests/README.md) for commands and fixture organization.

The complete local run on 2026-08-15 collected and passed 119 tests in 321.41 s
with a 2.56 GiB peak RSS; Ruff completed successfully in 0.03 s with a 35.5 MiB
peak RSS. The serial AlN fixture migration itself took 11.35 s and 375 MiB RSS;
the resulting FC3 reference tests took 7.34 s and 350 MiB RSS.

## Provenance and regeneration

Pinned upstream commits, licenses, hashes, versions, and regeneration commands are stored beside
each fixture under [`tests/reference`](../tests/reference/). The scripts in
[`reference_tools`](../reference_tools/README.md) regenerate those fixtures; MLFCS does not depend
on hiphive, phonopy, phono3py, pypolymlp, or symfc at base-package runtime.
