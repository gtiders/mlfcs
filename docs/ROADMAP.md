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
2. **Regularized/damped constrained fitting at high order.** The default unregularized
   `damping=0` path already uses a block-sparse null space, but the implicit projectors used by
   scaled group LASSO and nonzero damping still form a dense `C @ C.T`. Add automatic memory
   estimation and replace that projector for large constraint systems with a tested sparse method,
   such as sparse KKT/MINRES or an augmented-Lagrangian formulation. Selection must be automatic
   and preserve strict ASR and rotational residuals.

Items without an explicit design above remain open to implementation changes. All changes will be
introduced only with focused tests and documented numerical evidence.

3. **Rotational constraint conditioning bug.** The current hard FC1-FC2 boundary uses a
   structure-tolerance-aware numerical rank filter, while hiphive uses a ridge-regularized
   soft projection. These are different semantics: near-zero singular directions can be
   promoted to hard constraints or discarded depending on units and scaling. Replace the
   heuristic threshold with an explicit dimensionless conditioning policy, and provide a
   separate soft Huang/Born-Huang projection. Validate FC2, FC3, and FC4 against materialized
   physical IFC identities, including planar graphene and MoS2 cases.
