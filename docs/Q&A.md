# Q&A

## Can finite-difference reconstruction take a NumPy force array?

No. `FiniteDifference.reconstruct(structures)` takes an ordered sequence of ASE `Atoms`, with the force result stored on each structure. The geometry and atom order identify which generated displacement each force belongs to. For an external calculation, attach forces with `SinglePointCalculator` and preserve the sequence from `fd.displacements()`.

## Does `fd.evaluate(calculator)` reuse stored forces?

No. It requests fresh forces for every generated geometry. Use it when an ASE calculator should evaluate the structures. For forces computed by external DFT or another external program, attach each result to its corresponding ASE structure and call `reconstruct`.

## Can I use MACE, NEP, or an external DFT code?

Any ASE-compatible calculator can be passed to `fd.evaluate()` or to `SSCHA.run()`, provided it supports the input system. For fitting, calculate forces first and pass force-bearing ASE structures to `FitSystem.from_atoms`. External programs without an ASE calculator can still supply data by writing the generated structures, calculating forces externally, and attaching the read-back forces to ASE `Atoms`.

## Why must finite-difference input order be preserved?

Each position in the generated sequence corresponds to a particular cluster, sign combination, and displacement length. Reconstruction checks the geometry and atom sequence at every position; it does not infer a new ordering. Do not sort, deduplicate, or independently reorder the returned structures.

## Does one finite-difference object calculate multiple orders?

No. `FiniteDifference(mapping, order=..., disps=...)` handles one order. Multiple distinct displacement lengths are allowed and are extrapolated to zero displacement. To reconstruct multiple orders, create one finite-difference object for each order. A single `ClusterSpace` and `FitSystem` can instead fit multiple orders jointly.

## Does fitting calculate forces?

No. `FitSystem.from_atoms` reads forces already stored on ASE structures; it does not call their calculators. This lets the user choose DFT, an ASE machine-learning potential, or another force-generation workflow.

## What does the supercell mapping check?

`ClusterMap` connects a `ClusterSpace` to one explicit `Supercell`. `mapping.rank_info(order).require_full()` checks whether this supercell can distinguish the model parameters structurally. Run it before an expensive calculation. A full mapping rank does not guarantee that a particular fitting dataset has enough varied structures.

## Why does the default fit build a normal system?

`FitSystem` streams each structure into the sufficient statistics $A^T A$ and $A^T f$, so it can solve and merge fits without retaining all design rows. The default solver is column-scaled MINRES. If an algorithm needs the original equations, `FitData.from_atoms(...).arrays()` provides the design matrix and force vector for a direct least-squares solver; this retains more data in memory. See [fitting](fitting.md) for both paths.

## What if a fitted parameter is unobserved?

An exactly zero diagonal in the fit normal matrix means the corresponding design column is absent. The default solve raises `UnobservedParameterError`. Add informative structures, change the supercell or revise the model. Regularization cannot supply information that the training data do not contain.

## How do I choose and change masses?

Masses are taken from the ASE `Atoms` used to construct `ClusterSpace`. For custom isotope or site masses, set them on that ASE structure before creating the space, for example with `atoms.set_masses([...])`. The resulting masses flow through the model, mapping, and reciprocal calculations. There is currently no independent mass-override argument on `Harmonic`, `SCPH`, or `SSCHA`; a phonon calculation also requires the mass pattern to be compatible with the symmetry used for reciprocal-star reduction.

## Are ASR and rotational invariance applied during fitting?

No. They are explicit `ForceConstants` projections after fitting or finite-difference reconstruction. To impose both, apply ASR first and then rotational projection; the rotation step preserves the model's existing acoustic residual. The [force-constants guide](force-constants.md) explains the parameters and reports.

## How do I handle long-range dipole forces?

Use `Ewald` explicitly: subtract its force from every training frame before fitting the short-range model, then add `ewald.fc2` to the fitted FC2 array. The Ewald tensor satisfies ASR by construction. Its FC2 export does not include non-analytic LO-TO splitting; a downstream phonon calculation still needs its separate NAC data. See [long-range forces](long-range-forces.md) and the [NaCl example](../tutorial/NaCl/README.md).

## How do SCPH and SSCHA treat imaginary modes?

SCPH reports signed imaginary frequencies and can continue its static loop calculation using the magnitude of negative curvature in its covariance. SSCHA sampling needs a positive Gaussian trial FC2; it reports an unstable trial and can use Cartesian bootstrap only when `bootstrap_displacement` is supplied. The methods therefore have different stability requirements. See [SCPH](scph.md) and [SSCHA](sscha.md).

## How do temperature series proceed?

Both `SCPH.run_many()` and `SSCHA.run_many()` accept strictly ascending temperature sequences, execute from high to low, and return results in ascending order. SCPH carries a converged FC2 to the next lower temperature. SSCHA also warm-starts from the preceding result, and stops with `SSCHAContinuationError` if an intermediate temperature fails to converge.

## How do I save or export force constants?

Use `ForceConstants.save(path)` and `ForceConstants.load(path)` for native `.mlfcs` storage; only load native pickle files from trusted sources. Use `ForceConstants.write(path, mapping, format=..., order=...)` for model-based format export. For an already obtained compact FC2 array, use `write_phonopy(...)`. See [force constants](force-constants.md) for supported formats and orders.
