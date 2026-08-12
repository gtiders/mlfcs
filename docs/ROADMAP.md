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

No implementation choice for these items is committed yet. Changes will be introduced only with
focused tests and documented numerical evidence.
