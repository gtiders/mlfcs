# Translational and rotational sum rules

[中文](SUM_RULES_ZH.md) | English

MLFCS applies physical sum rules in the independent symmetry-orbit parameter space. They are
constraints on reconstructed force constants, distinct from the discrete rotations already used
to reduce tensors by crystal symmetry.

## Translational invariance

The acoustic sum rule is available at every supported order and is enabled by default:

```python
fc = calculation.reap(forces, acoustic_sum_rule=True)
```

MLFCS reports the maximum atomic-sum residual before and after projection:

```text
- Max drift of fc3: 2.3410000000e-03 -> 7.1200000000e-12 eV/angstrom^3
- ASR parameter correction: maximum=8.4100000000e-04 eV/angstrom^3, relative L2=1.7300000000e-03
```

The drift measures violation of the sum rule, whereas the correction measures how much the IFC
parameters were changed to enforce it. A small final drift does not imply a small correction, so
both values should be inspected when the cutoff support is incomplete. When translational and
rotational rules are enabled together, the correction is labelled as their joint projection.

## Harmonic rotational invariance

Rotational invariance is an adjacent-order hierarchy beginning with FC1--FC2. A finite-difference
calculation has no fitted FC1 unknown and treats its relaxed reference as FC1=0. The lowest identity
then reduces to the Born--Huang FC2 condition: an infinitesimal rigid rotation produces no harmonic
restoring force. It uses periodic minimum-image relative vectors and is disabled by default:

```python
fc2 = calculation.reap(
    forces,
    acoustic_sum_rule=True,
    rotational_sum_rule=True,
)
```

The translational and rotational matrices are stacked and projected in one sparse LSMR solve, so
one projection cannot undo the other. Both residuals are reported. `verbose=False` suppresses the
messages.

FC1=0 is a finite-difference boundary condition, not an estimate from the displaced forces. A
non-negligible reference force therefore indicates that the selected origin is inconsistent with
this boundary. The reference structure should be well relaxed.
Residual forces, stress, an undersized supercell,
or a cutoff support that is not sufficiently complete can make rotational projection large or
physically misleading.

## Higher orders

The current single-order API deliberately rejects `rotational_sum_rule=True` for orders above 2.
Rigorous higher-order rotational identities couple adjacent orders—for example FC3 to FC2 and FC4
to FC3. The separate `mlfcs.fitting` development API supplies this joint-order interface through
`rotational_invariance=2` or `3`. A Wick fit can produce a genuine Taylor FC1 at the selected
reference, so both modes impose the complete fitted FC1--FC2 identity rather than setting FC1 to
zero. They also include all represented adjacent-order identities. Mode 2 leaves only the
unrepresented upper boundary open;
mode 3 additionally sets that next-order contribution to zero. Because the fitter uses Wick
polynomials internally, it maps
the Taylor rotational system into Wick coordinates as `C_W = C_T @ T(Sigma)` before solving and
uses the same map for output. The finite-difference single-order restriction remains unchanged.
