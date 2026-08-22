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

`mixing` is numerical damping. An iteration is accepted as converged when the RMS frequency
change on the interpolation grid is below `tolerance` and all squared frequencies are non-negative.
This implementation is loop-only; it does not include the frequency-dependent bubble self-energy.
