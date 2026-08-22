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
4. **General periodic-image geometry.** The compatibility path currently preserves thirdorder's
   joint cluster test over the 27 neighboring supercell images. Investigate an adaptive image
   generator (potentially ASE `neighbor_list`) for strongly skewed cells or cutoffs that require
   more images, while retaining the joint multi-tail compatibility test. Do not replace the
   current convention until orbit counts, displacement order, and ShengBTE output compatibility
   have dedicated regression evidence. An initial comparison found no change for the KAsPt
   2x2x3 supercell through a 12 angstrom cutoff, but a deliberately non-reduced skewed cell needed
   translation `(-2, 1, 0)` and changed both FC2/FC3 orbit counts and the FC3 displacement set.

No implementation choice for these items is committed yet. Changes will be introduced only with
focused tests and documented numerical evidence.
