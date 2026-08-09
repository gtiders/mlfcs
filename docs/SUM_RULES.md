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
```

## Harmonic rotational invariance

For FC2, optional Born-Huang rotational sum rules impose zero restoring force for an
infinitesimal rigid rotation. They use periodic minimum-image relative vectors and are disabled by
default:

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

The reference structure should be well relaxed. Residual forces, stress, an undersized supercell,
or a cutoff support that is not sufficiently complete can make rotational projection large or
physically misleading.

## Higher orders

The current single-order API deliberately rejects `rotational_sum_rule=True` for orders above 2.
Rigorous higher-order rotational identities couple adjacent orders—for example FC3 to FC2 and FC4
to FC3—and therefore require a future joint-order constraint interface. MLFCS does not label an
order-local approximation as complete higher-order rotational invariance.
