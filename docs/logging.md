# Logging

MLFCS installs one package logger at import time. Its default level is INFO;
each record includes local time, level, module and message:

```text
2026-10-03 14:15:00 INFO mlfcs.fitting.system: Fit solve started: ...
```

Logs are flushed to the current Python stdout. File ownership and redirection
belong to the caller. There is no `stream`, file-path or format option:

```python
import logging
from mlfcs.log import configure

configure(level=logging.DEBUG)    # detailed diagnostics
configure(level=logging.WARNING)  # warnings and errors only
```

Repeated calls change the level without adding duplicate package handlers.
The root logger and user-installed handlers are left untouched.

## Capture a task

Use Bash to collect package records, ordinary prints, stderr and uncaught
tracebacks into one overwritten file:

```bash
python -u run.py > run.log 2>&1
```

To see output while saving it and preserve a failing Python exit status:

```bash
set -o pipefail
python -u run.py 2>&1 | tee run.log
```

Teaching fits have a separate requirement: every fit script itself overwrites
its local `fit.log` and captures stdout, stderr and traceback. These logs are
tracked case results. Each script owns its capture; there is no shared logging
wrapper. The package handler follows script-local stdout redirection, including
restoration after the file closes.

## Computation summaries

INFO records cover cluster construction, symmetry/candidates/orbits, supercell
mapping and folded rank, force-design compilation, fitting, finite differences,
harmonic meshes, SCPH, sum-rule projection and model save/load/export. Successful
completion records include relevant sizes, results and elapsed time. Fitting
and calculator evaluation report progress after the first item and every tenth
item; fitting iterables are consumed once. Timings include any JIT compilation.

The fit solver reports its algorithm, stopping settings, termination code and
iterations. The fit summary reports training force RMSE in eV/angstrom and
relative force error in physical units. Normal-equation summaries use the
stored sufficient statistics; they do not recover discarded individual forces.
If roundoff makes that diagnostic unavailable, a warning reports it without
invalidating a successful solve. INFO quality reporting evaluates one residual;
WARNING disables that additional work. Logging does not change fitted coefficients.

DEBUG adds workspace details. Logs do not compute or record model identity hashes.

## Exceptions

Scientific APIs raise their normal exceptions. They do not install a global
exception hook or log and re-raise the same failure at every layer. At a task
boundary, standard logging can record a caught failure:

```python
from mlfcs.log import get_logger

logger = get_logger("task")
try:
    main()
except Exception:
    logger.exception("Task failed")
    raise
```

Choose either that task-boundary recording or the uncaught traceback capture
when a single traceback is desired. Exceptions need no custom `.log()` method.
