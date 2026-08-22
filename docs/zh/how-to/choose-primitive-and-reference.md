---
title: 结构与后处理软件
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 结构与后处理软件

## 问题

primitive 必须由用户显式提供。参考超胞可以由整数 `3×3` 矩阵生成，也可以直接提供任意原子顺序的 结构。MLFCS 为每个参考原子保存 primitive site 标签和整数平移。 在开始计算前，尽可能从实际读取结果的软件取得结构，以避免 phonopy、phono3py、ShengBTE 和 ALAMODE 之间的约定差异。导出时可以提供严格等价的 primitive 或 supercell 表示，但不能改变 primitive 体积、 原子数、平移晶格，也不能改变 supercell 包含的原胞数量。 整数幺模换基只改变晶格矢量的坐标表示，不改变晶格或体积；它不是新的超胞。

## 操作步骤

先固定 primitive、reference 和目标输出，再记录本任务涉及的阶数、cutoff、body order、约束和单位。运行前验证所有输入结构与 reference 的晶格、原子标签和顺序一致。

## 结果检查

检查日志中的结构匹配、orbit/参数数量、秩、残差和异常警告。生成 IFC 后，应在明确的 target supercell 上检查数组或声子结果，而不是只确认文件成功写出。

## 为什么这样做

MLFCS 把物理 IFC 与具体文件排列分开。显式固定结构和验证目标表示，可以避免静默重排、周期镜像误判以及不同软件默认值造成的差异。

## 限制

本指南不替代电子结构收敛测试或后处理软件的参数收敛测试。长程静电、有限超胞 aliasing 和训练集覆盖不足也不能仅靠调整拟合参数修复。
