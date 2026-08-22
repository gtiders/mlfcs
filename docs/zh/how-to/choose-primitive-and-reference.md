---
title: 结构与后处理软件
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 结构与后处理软件

primitive 必须由用户显式提供。参考超胞可以由整数 `3×3` 矩阵生成，也可以直接提供任意原子顺序的
结构。MLFCS 为每个参考原子保存 primitive site 标签和整数平移。

在开始计算前，尽可能从实际读取结果的软件取得结构，以避免 phonopy、phono3py、ShengBTE 和 ALAMODE
之间的约定差异。导出时可以提供严格等价的 primitive 或 supercell 表示，但不能改变 primitive 体积、
原子数、平移晶格，也不能改变 supercell 包含的原胞数量。

整数幺模换基只改变晶格矢量的坐标表示，不改变晶格或体积；它不是新的超胞。
