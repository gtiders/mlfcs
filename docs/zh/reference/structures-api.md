---
title: 结构 API
audience:
  - developer
status: stable
code_verified: 4.0.0a5
---

# 结构 API

记录扩包、结构关系、显式对齐和周期索引。

~~~python
build_supercell(
    primitive: Atoms,
    supercell_matrix: object,
    *,
    symprec: float = 1e-5,
) -> Atoms

align_structures(
    reference: Atoms,
    atoms: Atoms,
    *,
    tolerance: float = 1e-5,
) -> tuple[Atoms, float]
~~~

`build_supercell` 接受长度三元组或整数 $3\times3$ 矩阵，并返回 phonopy old-style 原子顺序。`align_structures` 必须显式调用；拟合与有限差分不会静默重排快照。
