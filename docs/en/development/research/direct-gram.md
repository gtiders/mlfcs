---
title: Direct-Gram research
audience:
  - developer
status: research
code_verified: 4.0.0a4
---

# Direct-Gram and matrix-free design research

This research tested exact accumulation of $G=X^TX$ and $b=X^Ty$ without materializing the current streamed design batches. The evaluated interaction-streaming, feature-factorization, and finite-sample moment approaches did not reduce the dominant contraction cost without introducing parameter-squared work or larger intermediates, so the production fitting path remains unchanged.

## Decision

The current result is **No-Go**. The prototype is retained as a mathematical and performance boundary, not as a public API or alternative runtime path.
