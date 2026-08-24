---
title: I/O API
audience:
  - developer
status: stable
code_verified: 4.0.0a5
---

# I/O API

记录原生 HDF5 读取和带 target 要求的显式多格式 writer。

~~~python
read_hdf5(source: str | Path) -> ForceConstants

write_force_constants(
    force_constants: ForceConstants,
    target: str | Path,
    *,
    format: str,
    order: int | None = None,
    primitive: Atoms | None = None,
    supercell: Atoms | None = None,
) -> None
~~~

`format` 必须显式指定。原生 HDF5 不需要 target；外部稠密或格式受限输出可能需要经过验证的 primitive 或 supercell 视图。
