---
title: Force-constant API
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Force-constant API

Document pure force-constant data, sparse orders, materialization, and explicit target realization.

~~~python
realize_force_constants(
    force_constants: ForceConstants,
    reference: Atoms,
    *,
    primitive: Atoms | None = None,
) -> ForceConstants
~~~

`ForceConstants` stores available arrays, sparse exact-$R$ orders, metadata, and the structure relation needed by current operations. Target realization is an explicit function; writing and rotational correction are not methods on the data object.
