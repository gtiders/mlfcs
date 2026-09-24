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
