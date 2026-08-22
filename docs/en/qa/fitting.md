---
title: Fitting questions
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# Fitting questions

## Scope

Cover joint orders, one-reference restrictions, batching, covariance, rank, and fit diagnostics.

## How do I separate input errors from algorithm errors?

Validate structures, units, atom order, and training data, then compare IFCs or predicted forces in the same target representation.

## Why can different files describe the same physics?

Formats may use different block order, periodic representatives, and atom order. Align labels, translations, and tensor components before comparing.

## When should I change the setup?

Stop on structure mismatch, rank deficiency, abnormal residuals, or an unrepresentable target format; do not ignore the diagnostic.

## Where is the full explanation?

Definitions are in Concepts, derivations in Theory, tasks in How-to, and complete signatures in Reference.
