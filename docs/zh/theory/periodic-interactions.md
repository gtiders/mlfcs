---
title: 周期几何与 IFC
audience:
  - advanced
status: stable
code_verified: 4.0.0a6
---

# 周期几何与 IFC

`StructureRelation` 在 primitive 与一个具体 reference 之间建立映射，不依赖对角 `repeats` 或
cell-major 原子顺序。其基于 HNF 的商群只负责把 exact 晶格标签 realization 到有限 reference；
interaction 的发现则直接在无限 primitive 晶格中完成。

原生稀疏模型保存 lattice-labelled 条目：

```text
sites                        (K, order)
translations  (K, order - 1, 3)
tensors                      (K, 3, ..., 3)
```

这里的整数平移是 exact primitive 晶格矢量，不是有限超胞 residue。因此 Fourier 相位可直接使用
这些平移，同一组 IFC 也可展开到任意经过验证的整数超胞。residue reduction 只发生在有限计算视图
或输出边界；格式特有的周期镜像规则留在相应 writer 内部。
