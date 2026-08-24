---
title: Performance roadmap
audience:
  - developer
status: research
code_verified: 4.0.0a6
---

# Performance roadmap

## Current status

Separate measured bottlenecks and accepted optimizations from rejected geometry caching, unsafe segment buffers, and Direct-Gram No-Go results.

## Scope

Roadmap material is not a stable API. Equations, prototypes, and research conclusions must not be treated as implemented features.

## Entry criteria

Implementation requires a physical definition, data requirements, canonical-IFC interface, failure semantics, and a reproducible baseline. Experimental compatibility branches are not permanent.

## Acceptance

Require analytic tests, structure/symmetry regressions, material cases, and resource measurements. Classify changed results before updating baselines.

## Non-goals

Do not expand features by silently averaging, projecting, or regularizing unidentifiable information. No-Go findings remain design evidence.
