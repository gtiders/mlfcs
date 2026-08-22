# Wick contractions and Taylor cross-order leakage

[中文](WICK_TAYLOR_CROSS_ORDER_ZH.md) | English

## The question

MLFCS fits FC2--FCn jointly in covariance-orthogonalized Wick polynomials and converts the result
to ordinary Taylor force constants at the reference structure. Including FC4 therefore produces

```text
FC2_T = FC2_W - 1/2 FC4_W:Sigma + 1/8 FC6_W:Sigma:Sigma - ...
```

This resembles a higher order entering a lower order, but it is not the same phenomenon as
cross-order leakage in an ordinary Taylor regression. The former is an exact deterministic basis
change; the latter is ambiguity or bias in an estimate made from finite data.

## Why FC4 enters Taylor FC2

In one dimension the fourth Wick polynomial is

```text
H4(x; sigma^2) = x^4 - 6 sigma^2 x^2 + 3 sigma^4.
```

With the usual factorial convention,

```text
E = FC2_W H2 / 2! + FC4_W H4 / 4! + ...,
```

expansion into ordinary displacement powers gives

```text
FC2_T = FC2_W - FC4_W sigma^2 / 2.
```

In multiple dimensions, `sigma^2` becomes the supercell displacement covariance `Sigma` and the
product becomes a contraction over atom and Cartesian indices. FC6 similarly contracts into FC4
and FC2; odd orders form their own hierarchy, including FC3 contracting into FC1.

This is not a second fit, an iterative correction, or an arbitrary transfer of physical FC4 into
FC2. Wick and Taylor coefficients describe the same truncated polynomial in different
coordinates, and its energies and forces are unchanged by the conversion. Physical IFCs are
Taylor derivatives at the reference structure, so public `ForceConstants` and external writers
must contain the converted Taylor tensors.

## Genuine Taylor cross-order leakage

Taylor force features resemble successive powers `x`, `x^2`, and `x^3`. Their columns can be
strongly correlated in a finite sample. If the true force contains a cubic-displacement term but
the fit contains only the linear feature, the omitted FC4 contribution projects into FC2. Even a
joint FC2+FC4 fit can trade the orders against each other when data are finite or noisy, the design
is ill-conditioned, interactions are truncated, or regularization is applied. This statistical
effect is the usual cross-order leakage.

Taylor coordinates do not inevitably cause leakage. A complete, full-rank model fitted exactly to
noise-free data can recover unique Taylor coefficients. The practical problem is statistical
identifiability, not the algebraic definition of a Taylor expansion.

Wick features make different degrees orthogonal for an ideal Gaussian distribution consistent
with `Sigma`, reducing same-parity column correlation and finite-sample absorption. For finite,
non-Gaussian, or asymmetric data this is mitigation, not a guarantee. Wick coordinates also
cannot restore physical interactions excluded by the maximum order, spatial cutoff, or body order.

## Similarities and differences

| Question | Wick-to-Taylor contraction | Taylor cross-order leakage |
|---|---|---|
| Nature | Exact coordinate change of one polynomial | Regression ambiguity, variance, or bias |
| Present with perfect data | Yes | Not for a complete full-rank model |
| Dependence on `Sigma` | Explicit; `Sigma` defines the Wick basis | Indirect through sampling and column correlation |
| Changes the predicted polynomial | No | Usually changes the estimated physical polynomial |
| Why FC2 may change after adding FC4 | Deterministic conversion to Taylor FC2 | Previously omitted effects may be released from estimated FC2 |
| Removed by more data | No, and it should not be removed | Often reduced |

Both describe a relationship between lower and higher orders over a finite-temperature displacement
distribution, and both can make FC2 change visibly when FC4 is included. The distinction is that a
Wick contraction restores reference-point derivatives from orthogonal statistical coordinates,
whereas leakage means the data have not separated different Taylor contributions robustly.

## Diagnosing which effect is present

1. With one training set and fixed `Sigma`, verify
   `FC2_T = FC2_W - 1/2 FC4_W:Sigma + ...`. A difference explained by this identity is conversion,
   not leakage.
2. Compare snapshot forces from the complete model before and after conversion. Agreement to
   numerical tolerance shows that only the representation changed.
3. Vary the snapshots, displacement amplitude, and maximum order, then monitor final Taylor FC2,
   validation error, and design conditioning. Drift beyond statistical uncertainty indicates
   truncation or identifiability risk.
4. Do not compare `FC2_W` directly with finite-difference, ALAMODE, or phonopy FC2. Those quantities
   are Taylor derivatives and must be compared with `FC2_T`.
