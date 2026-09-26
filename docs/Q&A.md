# Q&A

## Can I pass a NumPy force array to finite-difference reconstruction?

No. `FiniteDifference.reconstruct` consumes an ordered sequence of ASE `Atoms`, each with its force result attached. Geometry, atom order, and force data form one record; a bare array cannot certify which displacement it belongs to.

## Does `fd.evaluate(calculator)` use cached forces?

It deliberately requests a fresh force calculation for every generated structure and stores the result on the returned structures. Use this when the calculator should evaluate the current geometry. If forces came from an external program, attach them as stored ASE results and call `reconstruct` directly.

## Can I use MACE or another ASE calculator?

Yes. MLFCS does not own the calculator. Any ASE-compatible calculator can be used with `fd.evaluate(calculator)` if it supports the species and conditions in the input structures. For a force-fitting workflow, calculate forces with your chosen calculator first and provide the resulting ASE structures to `FitSystem.from_atoms`.

## Why is finite-difference input order strict?

The displacement index encodes which mixed derivative is measured. Keep the order returned by `displacements()` through external evaluation. The reconstruction checks frame geometry and atom sequence and refuses mismatches; it does not infer a new ordering.

## Can a finite-difference object be pickled and reloaded?

The supported workflow is to regenerate the deterministic sequence from the same primitive model, cluster space, explicit supercell, and mapping, then pass the evaluated ASE frames in their original order. MLFCS does not define a serialized experiment-plan format. Persist the ASE structures and forces with a suitable data format when moving calculations between programs.

## Does one finite-difference object calculate several orders?

No. One `FiniteDifference` object handles one order. For joint multi-order fitting, use one `ClusterSpace` with per-order cutoffs and body-order limits, then build one `FitSystem`.

## Does fitting calculate forces?

No. `FitSystem.from_atoms` only reads forces already stored on each ASE `Atoms`; it does not call the attached calculator. This keeps force generation under the user's control and works with external electronic-structure or machine-learning calculations.

## What is the role of the supercell?

The primitive cluster space defines which interactions and symmetry-reduced parameters are in the model. The explicit supercell maps those primitive interactions onto the training atom list. Its size and shape determine whether the requested parameters can be distinguished. Check `mapping.rank_info(...).require_full()` before an expensive calculation.

## Why does fitting construct a normal system?

The optimized fit path accumulates the sufficient statistics `A.T @ A` and `A.T @ f` while streaming training structures. It avoids retaining a potentially huge design matrix and allows compatible systems to be merged or reused. The default solver is column-scaled MINRES; it is not a batch-gradient neural-network optimizer.

## What if a parameter is unobserved?

An exactly zero normal-matrix diagonal identifies a parameter absent from the training design. The default solver rejects this with a named error. Add structures or displacements that excite the missing direction, use a more informative supercell, or revise the model. Regularization cannot create information that is absent from the data.

## Are ASR and rotational conditions part of fitting?

No. Fit first, then apply the explicit force-constant post-processing projection if desired. This keeps the linear force fit separate from the choice of physical constraints. Review the projection report and its effect on the force constants.

## How do I save or export the result?

Use `ForceConstants.save(path)` for native trusted-pickle storage and `ForceConstants.load(path)` to read it back. For interoperable output, use `ForceConstants.write(path, mapping, format=..., order=...)`. Native pickle files must come from trusted sources.

## How do I set isotope or other per-site masses?

Pass a list in primitive-site order, in atomic mass units: `PrimitiveCell.from_atoms(atoms, symprec=1e-5, masses=[28.0, 29.0])`. A NumPy array also works. With `masses=None`, ASE's default elemental masses are used; custom masses stored on the input ASE `Atoms` are not implicitly adopted. `primitive.with_masses([...])` returns a new immutable primitive. Symmetry-related sites must have equal masses for reciprocal-space reduction.

Native `.mlfcs` files now use version 2 and preserve these masses. Version 1 files are rejected; regenerate them with the fitting or finite-difference task that produced them. Changing masses leaves the cluster-space geometry and parameter layout unchanged, but changes the saved force-constant model's fingerprint and its phonon frequencies.

## How do I run stochastic self-consistent harmonic fitting?

Build an FC2-only `ClusterMap` for the explicit supercell, then pass it to `SSCHA(mapping, mesh, initial=fc2, seed=42)`. The mesh must be that supercell's reciprocal grid; for a diagonal 2×2×2 supercell, use `mesh=(2, 2, 2)`. Call `solver.run(300, calculator, pairs=128)` with any suitable ASE `Calculator`. SSCHA itself generates paired displacements, requests fresh forces, and fits FC2. It does not accept force arrays or pre-evaluated structures. `run_many([0, 100, 300], calculator)` solves from high to low temperature and returns results in ascending order.

An imaginary frequency in the input FC2 is meaningful, but it cannot define a harmonic Gaussian sampling distribution. Supply a stable trial FC2 or set a physical `bootstrap_displacement` for a Cartesian initialization. Negative modes are never replaced by their absolute values or silently omitted. The returned `status` distinguishes convergence, sample limits, iteration limits, and an unstable update. This first implementation fixes both the cell and mean atomic positions; it is not a full structural or free-energy-Hessian optimization.

If a Gaussian displacement crosses the supercell minimum-image boundary, the run stops before calculating its forces. Increase the supercell or choose a better stable trial; folding that displacement would silently change the FC2 fitting data.
