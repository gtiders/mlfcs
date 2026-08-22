---
title: 力常数 API
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# 力常数 API

记录纯力常数数据、稀疏阶、materialization 和显式 target realization。

~~~python
realize_force_constants(
    force_constants: ForceConstants,
    reference: Atoms,
    *,
    primitive: Atoms | None = None,
) -> ForceConstants
~~~

`ForceConstants` 保存已有数组、稀疏 exact-$R$ 阶、metadata 和当前操作所需结构关系。目标 realization 是显式函数；写出和旋转修正不是数据对象的方法。
