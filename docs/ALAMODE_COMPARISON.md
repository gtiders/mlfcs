# MLFCS fitting versus ALAMODE

This document compares the force-only `mlfcs.fitting` backend with ALAMODE's IFC optimization,
not the complete scope of either project. ALAMODE includes a mature phonon, anharmonic phonon,
transport, and SCPH ecosystem; MLFCS also provides a separate finite-difference `sow()` / `reap()`
path and general force-constant I/O.

Both solve the same linear force model, `F = A Φ`, using symmetry-reduced IFC parameters. MLFCS
does not replace this regression with a neural network. Its principal architectural distinction is
that `A` can be evaluated matrix-free or reduced batchwise to streamed Gram sufficient statistics,
whereas ALAMODE supports dense SVD/QR and explicit sparse Eigen solvers.

## Summary

| Aspect | MLFCS current backend | ALAMODE `alm` |
|---|---|---|
| Interface | Python API and ASE `Atoms` | Input-file and DFSET workflow |
| Current fitted orders | Consecutive joint FC2 through arbitrary configured order; FC2--FC4 tested | Mature second and higher orders |
| Basis | Covariance-orthogonalized Wick features | Ordinary Taylor displacement powers |
| Design matrix | Matrix-free or streamed `AᵀA`, never retained in full | Dense or explicit sparse matrix |
| Solvers | LSMR; streamed Gram with implicit constraint null space | OLS using SVD/QR or Eigen sparse solvers |
| Scaling | Per-parameter column-norm preconditioning | Optional column standardization for elastic net or displacement scaling |
| Regularization | LSMR damping/implicit early regularization | OLS, elastic net, adaptive LASSO, CV, and debiasing |
| Constraints | Translational/rotational relations embedded through an implicit null-space projector | Mature translational and rotational constraint relations |
| Acceleration | JAX JIT and bounded CPU/GPU batches | C++, Eigen, MPI, and sparse linear algebra |
| Ecosystem | ASE and phonopy/ShengBTE/HDF5/NumPy output | ALAMODE XML and comprehensive downstream analysis |

## MLFCS strengths

- The full `N_force × N_parameter` matrix is never resident in memory. Increasing the number of
  snapshots primarily increases operator traversal time rather than matrix storage.
- `batch_size` is a resource control, not stochastic optimization: every LSMR iteration still uses
  the complete dataset through exact accumulated `A·x` and `Aᵀ·r` operations.
- Per-parameter column scaling handles the very different `u`, `u²`, and higher feature scales
  without order-specific learning rates.
- Wick features reduce correlations between adjacent polynomial orders. This mitigates, but does
  not eliminate, lower-order absorption of omitted higher-order effects.
- The completed Wick hierarchy is converted exactly to ordinary Taylor IFCs at the
  `ForceConstants` boundary. Same-parity contractions such as
  `FC2_T = FC2_W - 1/2 FC4_W:Sigma` are an algebraic change of coordinates, not another fit.
- ASE input, JAX execution, and the common `ForceConstants` model make calculator integration and
  phonopy, ShengBTE, or HDF5 export direct Python operations.

ALAMODE also has a mature column-standardization option for elastic net. MLFCS differs by using
column scaling as the default numerical preconditioner for matrix-free LSMR.

## Current MLFCS limitations

- FC4 is now exercised end to end, but ALAMODE has much longer-established FC4+ elastic-net and
  adaptive-LASSO workflows and broader validation evidence.
- MLFCS lacks regularization paths, k-fold cross-validation, automatic regularization selection,
  adaptive LASSO, and debiased OLS.
- Matrix-free LSMR estimates column norms stochastically; the streamed-Gram backend obtains exact
  norms and retains an internal failure-recovery checkpoint for large systems.
- Matrix-free iterations repeatedly traverse all snapshots. For a small problem that fits in RAM,
  one dense construction followed by SVD or QR may be faster and gives clearer rank diagnostics.
- Constraint relations are solved in an implicit numerical null space, including redundant rows;
  ALAMODE's constraint machinery nevertheless has broader long-term production validation.
- MLFCS does not replace ALAMODE's integrated SCPH, transport, and anharmonic-analysis ecosystem.

## Current numerical evidence

For the repository KAsPt test with a 2x2x3 supercell, 150 random snapshots, unrestricted FC2, and
a 12-bohr FC3 cutoff, the matrix-free problem has 48,600 force equations and 5,053 parameters.
LSMR converged in 32 iterations with 4.623205% training and 5.080075% validation relative force
errors. Peak resident memory was approximately 2.78 GiB.

For the same material with all 250 snapshots and a joint FC2+FC3+FC4 model (maximum FC2 range,
12-bohr FC3 and 8-bohr FC4 cutoffs), the problem contains 90,000 force equations and 7,756
irreducible parameters. The streamed-Gram CPU backend built its sufficient statistics in 590.00 s
and converged after 1,405 projected-CG iterations. The complete run took 719.81 s, peaked at about
4.87 GiB RSS, reached 5.954683% relative force error, and left a 4.07e-15 joint ASR residual. This
is an implementation benchmark, not a performance comparison with ALAMODE.

For this FC2--FC4 fit, Wick-to-Taylor conversion changes FC2 by 3.274% in relative L2 norm and
leaves FC3 and FC4 numerically unchanged. After periodic and permutation canonicalization against
the supplied ALAMODE text files, FC3 differs by 7.413% in relative L2 norm with cosine similarity
0.997350. FC4 differs by 52.015% on the common support; MLFCS also contains 54 additional
canonical FC4 clusters. These are numerical diagnostics, not a performance ranking.

## Choosing a tool

MLFCS is attractive for ASE-native data, matrix-free FC2--FCn fitting, Python calculator workflows,
and direct cross-format export. ALAMODE is preferable today for mature FC4+ sparse regression,
systematic regularization and cross-validation, fully integrated invariance constraints, and its
complete downstream anharmonic-phonon ecosystem.

The tools can also be complementary. Any numerical comparison must align the reference geometry,
atom order, units, cutoffs, periodic-image support, constraints, and train/validation partition.
