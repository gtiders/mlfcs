# Roadmap

[中文](ROADMAP_ZH.md) | English

Development is currently paused at the present `dev` implementation. The next milestone will
focus on two related tasks before further feature work:

1. **Full physical-validity testing.** Build a reproducible validation matrix covering held-out
   force prediction, symmetry and sum-rule residuals, order-by-order stability, phonon
   observables, and converged thermal-transport calculations. Comparisons against independent
   implementations will use matched supercells, interaction support, body order, conventions,
   and numerical settings. Agreement between two fitted IFC files alone will not be treated as
   physical ground truth.
2. **High-order irreducible-set scalability.** Diagnose and remove the combinatorial explosion in
   irreducible-cluster enumeration that currently makes sixth-order calculations impractical.
   The work will target earlier symmetry pruning, canonical incremental construction, bounded
   intermediate storage, and reproducible scaling measurements without changing the established
   FC2--FC4 results.
3. **Sparse constrained fitting at high order.** The current projected-Gram solver keeps the force
   design matrix streamed, but its null-space projector forms a dense `C @ C.T`. Add automatic
   memory estimation and replace that projector for large constraint systems with a tested sparse
   constrained method, such as a sparse KKT/MINRES or augmented-Lagrangian formulation. The
   selection must be automatic and must preserve strict ASR and rotational residuals.
4. **Periodic geometry stress testing.** General MIC and reduced-lattice degenerate-image handling
   now form the calculation core. Extend the validation matrix with larger skewed cells, boundary
   degeneracies, and additional external readers; ALAMODE's fixed 27-image encoding remains a
   format-level limitation and must keep rejecting non-representable geometry.
5. **Freeze previously calculated IFC orders during fitting.** Allow a native HDF5 v2 result to
   provide fixed low-order Taylor IFCs while fitting the remaining consecutive orders, initially
   through a constructor-level API such as `fixed={2: read_hdf5("fc2.h5")}`. This must be an exact
   affine constrained fit, not a post-fit overwrite: with Wick-basis fitting, FC4 contracts into
   Taylor FC2, and rotational identities couple adjacent orders. Build the combined system
   `E p = b`, obtain `p = p0 + Z q`, subtract the fixed contribution `A p0` from the force target,
   and solve only for `q`. Before fitting, strictly validate primitive/reference equivalence,
   lattice-labelled support, cutoff/body-order compatibility, orbit-pivot reconstruction, and the
   feasibility of requested ASR and rotational constraints; never silently project the supplied
   fixed IFCs. Return the fixed and fitted orders in one physical `ForceConstants` result, record
   their provenance, and include the fixed tensors plus affine map in the persistent Gram-cache
   fingerprint. The first implementation should reject scaled group LASSO with frozen orders
   until orbit-group regularization is defined correctly in affine coordinates.

Items without an explicit design above remain open to implementation changes. All changes will be
introduced only with focused tests and documented numerical evidence.
