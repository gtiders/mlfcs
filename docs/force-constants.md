# Force constants: model, constraints, and files

`ForceConstants` is the primitive-cell model produced by finite differences or fitting. It binds one or more order-specific parameter blocks to the `ClusterSpace` that defines their physical meaning.

## Construct and inspect a model

Typical construction uses the result objects from the two workflows:

```python
fc_fd = fd.reconstruct(evaluated_structures)
fc_fit = system.force_constants(parameters)
```

The direct constructor is `ForceConstants(space, coefficients)`, where `coefficients` maps each included integer order to a finite packed coefficient vector of the size defined by that order's parameter block. `orders`, `fingerprint`, `parameters(orders=None)`, and `coefficients` expose model contents. An absent order is not treated as a zero-valued order.

Models containing disjoint orders on the same space can be combined:

```python
combined = ForceConstants.combine((fc2, fc3))
```

`save(file)` writes the native `.mlfcs` representation; `ForceConstants.load(file)` reads it. The native representation uses pickle and should only be loaded from trusted sources.

## Get a supercell tensor

`model.get(order, mapping)` returns the chosen order as an unfiltered, primitive-first compact tensor. Its shape is `(n_primitive, n_supercell, ..., 3, 3, ...)`: there is one supercell atom axis for each index after the first primitive atom, and one Cartesian axis for each tensor index.

```python
fc2_array = combined.get(2, mapping)
```

The requested order must be present in the model, and the mapping must correspond to the same cluster space. `get()` supports each order represented by the model; format-specific array writing is narrower than tensor access.

## Acoustic sum rule (ASR)

Translational invariance requires that a uniform displacement does not change the force. The corresponding acoustic sum rule constrains the sum of force constants over translated atoms; for higher-order tensors the translational constraint applies to each interaction index. `ForceConstants.enforce_asr()` projects the chosen model orders onto these linear constraints.

```python
result = combined.enforce_asr(orders=(2, 3), rtol=1e-10)
model = result.force_constants
fc2_report = result.report(2)
print(fc2_report.relative_before, fc2_report.relative_after)
```

The signature is `enforce_asr(*, orders=None, rtol=1e-10)`. `orders=None` selects all orders present in the model; otherwise `orders` must be unique and ascending. `rtol` is the target relative constraint residual and must be positive. The method returns a new `ForceConstants` model and an `ASRResult`; it does not mutate the input.

Each `ASRReport` contains `order`, `equations`, `parameters`, `residual_before`, `residual_after`, `relative_before`, `relative_after`, `correction_norm`, `relative_correction`, and `iterations`. `ASRResult.report(order)` retrieves the report for one projected order.

## Rotational invariance

Rotational constraints enforce zero force-constant moments associated with infinitesimal rotations. The API acts on FC2 and separates the first-moment Born-Huang condition from Huang's second-moment condition. Huang's condition is appropriate for a stress-free equilibrium structure, so it is off by default.

```python
result = model.enforce_rotation(born_huang=True, huang=False)
model = result.force_constants
print(result.retained_rank, result.relative_before, result.relative_after)
```

The signature is `enforce_rotation(*, born_huang=True, huang=False, rank_rtol=None)`. Select at least one of `born_huang` and `huang`. `rank_rtol=None` derives the singular-value rank cutoff from the observed geometry residual; a supplied value in `[0, 1]` sets the relative cutoff directly. This controls which rotational constraint directions are numerically resolvable for the actual Cartesian geometry.

`RotationResult` returns the projected `force_constants` and records the selected conditions, `length_scale`, `equations`, acoustic/rotational residuals before and after, `relative_before`, `relative_after`, correction norms, `retained_rank`, rank cutoff and its automatic/manual status, retained/discarded singular-value bounds, and the geometry and orthogonality residuals. Born-Huang and Huang residual fields are `None` when that condition was not selected.

Rotational projection does not apply ASR and does not alter the input model's acoustic residual: it restricts the correction itself to the acoustic null space. To impose both, apply ASR first and then rotational projection; this lets the rotation step preserve the ASR-satisfying model. The returned reports expose the residuals so the effect of each projection can be inspected.

## Export model orders

`model.write(file, mapping=None, *, format, order, threshold=1e-8)` writes one order. `order` is required. `threshold` is an absolute component threshold applied during external-format expansion. `mapping` identifies the target supercell for formats that require one.

| `format` | Supported order | Mapping |
| --- | --- | --- |
| `phonopy_text` | FC2 | Required |
| `phonopy_hdf5` | FC2 | Required |
| `phono3py_hdf5` | FC3 | Required |
| `shengbte` | FC3 or FC4 | Required |
| `tdep` | FC2, FC3, or FC4 | Not required; if supplied, it is checked against the model |

```python
model.write("fc2-phonopy.hdf5", mapping, format="phonopy_hdf5", order=2)
```

The standalone `write_phonopy(file, array, mapping, *, format, threshold=1e-8)` writes an already obtained FC2 array in `phonopy_text` or `phonopy_hdf5` format. Its array is expected to match the primitive-first shape returned by `get(2, mapping)`.

## Long-range forces

Polar crystals can have slowly decaying dipole-dipole force constants. The `Ewald` calculator provides an explicit harmonic long-range component. The force-subtraction and FC2-reassembly workflow is described in [Long-range forces](long-range-forces.md); the [NaCl tutorial](../tutorial/NaCl/README.md) compares direct fitting with this correction.
