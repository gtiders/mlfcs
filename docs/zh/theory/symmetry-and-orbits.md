---
title: 对称性与轨道
audience:
  - advanced
status: stable
code_verified: 4.0.0a4
---

# 对称性与轨道

## 动机

说明空间群对 primitive site 和整数平移的仿射作用、指标置换、稳定子与不变 Cartesian 张量基。

## 数学对象

本节讨论的量定义在整个晶体的位移空间和周期晶格上。原子标签、Cartesian 分量与整数平移必须同时保留；只保留距离或超胞原子号通常不足以确定物理 interaction。

## 在 MLFCS 中的实现

MLFCS 将理论对象拆成结构映射、interaction/orbit、参数化和物理 IFC。计算过程可以使用约化参数，但输出必须恢复为带明确结构和平移标签的 Taylor 力常数。

## 数值注意事项

有限超胞、截断、浮点容差和矩阵秩都会影响可恢复信息。对称约束可以消除冗余，却不能创造训练数据中不存在的独立观测。

## 验证方式

实现应通过解析小模型、对称变换、原子重排、非对角超胞和真实材料案例交叉验证；内部 canonical ordering 改变不应改变最终物理量。
