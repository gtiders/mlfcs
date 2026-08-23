---
title: 能力与边界
audience:
  - beginner
status: stable
code_verified: 4.0.0a4
---

# 能力与边界

本页按照当前公共 API 记录稳定、实验、计划和未支持能力。功能状态是一项维护契约，必须随代码修改同步更新。

## 稳定能力

- 显式 primitive/reference 结构关系、HNF 周期索引、任意原子顺序和非对角整数超胞。
- 从 FC2 到已支持高阶的对称约化有限差分和仅力数据拟合。
- Taylor 输出、Wick 拟合坐标、平移约束和 FC2 Born–Huang/Huang 修正。
- 稀疏原生 HDF5 v3、target-supercell realization 和已记录的外部 writer。

## 实验能力

- FC4 loop SCPH。
- 随机有效谐波 SSCHA。

这些工作流返回完整有效 FC2 对象，但必须显式检查收敛性和物理结果。

## 计划能力

- 短程拟合前的长程偶极力扣除。
- 有限差分 signed-displacement 对称约化。

## 不支持

- 不同超胞数据联合拟合。
- 读取旧原生 HDF5 schema。
- 导出时整体 Cartesian 刚性旋转。
- 显式 FC3 bubble 自能或多极静电修正。
