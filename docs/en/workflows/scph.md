# Loop-SCPH

`LoopSCPH` applies the static quartic loop correction to FC2 and returns a temperature-dependent
effective FC2. FC2 and FC4 are separate `ForceConstants` objects and must describe the same
primitive/reference frame.

```python
result = LoopSCPH(
    fc2=fc2, fc4=fc4, temperature=600,
    interpolation_mesh=(3, 3, 3), scph_mesh=(6, 6, 6),
    mixing=0.1, tolerance=1e-10, max_iterations=100,
).run()
```

`mixing` is covariance under-relaxation. An iteration is accepted as converged when the RMS frequency
change on the interpolation grid is below `tolerance`. Imaginary frequencies are retained as a
physical diagnostic and do not add a separate stopping condition. This implementation is loop-only;
it does not include the frequency-dependent bubble self-energy.

For several temperatures, use temperature continuation. Temperatures are evaluated in ascending
order, and the effective FC2 from one temperature initializes the next one.

```python
results = LoopSCPH(
    fc2=fc2, fc4=fc4, temperature=range(300, 901, 300),
    interpolation_mesh=(3, 3, 3), scph_mesh=(6, 6, 6),
    max_iterations=100,
).run()
```
