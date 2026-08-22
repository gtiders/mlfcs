---
title: ShengBTE
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# ShengBTE

ShengBTE FC3 和 FC4 文本是按 block 组织的稠密 view。writer 先验证目标参考顺序和平移晶格，再把稀疏
lattice-labelled IFC 展开到目标顺序；不会改变物理支撑或选择新的原胞。

任意 q 点插值时，周期镜像选择属于通用几何层；核心 IFC 模型不会使用固定的 27 镜像盒子猜测它。
