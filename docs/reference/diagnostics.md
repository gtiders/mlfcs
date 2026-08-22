# Diagnostics

When a result is unexpected, check the structure relation first: primitive atom count, reference
atom order, supercell matrix, and maximum mapping residual. Then check cutoff shell, force units,
ASR residual, and the target writer's required supercell.

For SCPH, inspect `result.history`, the final RMS frequency change, and negative squared
frequencies. A non-converged effective FC2 is diagnostic output, not a production result.
