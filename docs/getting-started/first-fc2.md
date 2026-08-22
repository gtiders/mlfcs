# First FC2

Use `examples/basic_fc2.py` for the smallest calculator-backed calculation:

```bash
uv run python examples/basic_fc2.py
```

The output is a native HDF5 force-constant file. Use the [format guides](../formats/index.md)
to create a downstream representation. Dense materialization is explicit; high-order native
files remain sparse.
