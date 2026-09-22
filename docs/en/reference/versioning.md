---
title: Versioning Policy
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# Versioning policy

`mlfcs.__version__` reports the installed version. The project is still in alpha development, so
intentional breaking API changes are recorded directly in the changelog rather than hidden behind
compatibility aliases.

The stable boundary consists of the top-level `mlfcs.__all__`, advanced result objects explicitly
listed in this reference, documented external formats, and native HDF5 schema v4. Interaction
internals, packed design matrices, constraint matrices, and underscore-prefixed names are
implementation details.

The native reader accepts only schema v4. It does not guess the meaning of v1, v2, or v3 fields.
Future schema changes must use a new version and either an explicit migration tool or a clear
rejection.

The `code_verified` front-matter field identifies the source release against which a page was last
checked; it is not a promise that future releases remain compatible.
