---
title: 单位与张量分量
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 单位与张量分量

## 表示边界

定义长度、力、IFC 单位、Cartesian 分量顺序、阶乘约定和外部格式转换。

## 转换前必须确认

明确 source primitive、source reference、target structure、原子顺序、单位和张量分量约定。所有转换先验证，失败时不应留下部分写出的文件。

## 转换过程

先从 canonical primitive IFC 构造目标超胞上的 realization，再根据目标格式进行 folding、稠密化、block 排列或镜像编码。格式 writer 不重新选择 primitive，也不重新运行 orbit 枚举。

## 验证

优先执行往返或第三方实际读取测试，并在同一结构表示上比较数值。文件大小、行序和周期代表不同本身不构成错误。

## 无法表达的情况

目标格式若不能表示所需阶数、平移、镜像或结构关系，writer 应明确拒绝；不得通过平均、投影、扩包或改原胞来静默改变物理模型。
