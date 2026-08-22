# Loop-SCPH

`LoopSCPH` applies the static quartic loop correction to FC2 and returns a temperature-dependent
effective FC2. FC2 and FC4 are separate `ForceConstants` objects and must describe the same
primitive/reference frame.

```python
result = LoopSCPH(
    fc2=fc2, fc4=fc4, temperature=600,
    interpolation_multiplier=1, scph_multiplier=2,
    mixing=0.1, tolerance=1e-10, max_iterations=100,
).run()
```

`result.force_constants` is a normal, FC2-only `ForceConstants` object.  It
can therefore be written through the ordinary export API without a
SCPH-specific conversion:

```python
write_force_constants(result.force_constants, "scph.h5", format="hdf5")
write_force_constants(result.force_constants, "FORCE_CONSTANTS_SCPH", format="phonopy")
write_force_constants(result.force_constants, "force_constants.xml", format="alamode")
```

The q grids are reciprocal quotients of integer multiples of the reference
supercell matrix. `interpolation_multiplier` controls the reported frequency
grid and `scph_multiplier` controls the loop integration grid; the latter must
be an integer multiple of the former.

`mixing` is covariance under-relaxation. An iteration is accepted as converged when the RMS frequency
change on the interpolation grid is below `tolerance`. Imaginary frequencies are retained as a
physical diagnostic and do not add a separate stopping condition. This implementation is loop-only;
it does not include the frequency-dependent bubble self-energy.

For several temperatures, use temperature continuation. Temperatures are evaluated in ascending
order, and the effective FC2 from one temperature initializes the next one.

```python
results = LoopSCPH(
    fc2=fc2, fc4=fc4, temperature=[300, 600, 900],
    interpolation_multiplier=1, scph_multiplier=2,
    max_iterations=100,
).run()
```
