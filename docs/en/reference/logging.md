---
title: Logging
audience:
  - user
status: stable
code_verified: 4.0.0a6
---

# Logging

MLFCS reports its principal physical and numerical state through the standard Python logger named
`mlfcs`. By default, messages at `INFO` and above are written to stdout. The package does not
configure the root logger and does not redirect uncaught exception tracebacks from stderr.

To enable implementation-level details such as batches, timing, and rank tolerances:

```python
import logging

logging.getLogger("mlfcs").setLevel(logging.DEBUG)
```

Use the standard logging API to add filters or replace handlers. Public MLFCS workflows do not
accept `verbose`, `log_level`, or reporter callback parameters. Invalid states raise exceptions;
warnings are reserved for legal computations whose consequences require attention, such as an
explicit imaginary-mode policy, displacement clipping, or a returned unconverged iteration.
