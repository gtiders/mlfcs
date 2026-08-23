---
title: 为什么需要 MLFCS
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 为什么需要 MLFCS

## 定位与范围

本页把“为什么需要 MLFCS”放回 MLFCS 的完整工作流中，说明它解决什么问题、与哪些模块相连，以及哪些工作明确不由它承担。高阶力常数面临团簇组合增长、张量对称性、存储压力与 Taylor 阶间误差耦合。MLFCS 以统一的稀疏、对称感知表示连接有限差分、拟合、约束和导出。

## 如何使用本页

先确认输入结构、单位和目标后处理程序，再沿页面给出的入口进入 Concepts、Tutorials 或 How-to。不要把概览页当作参数参考；可执行签名以 Reference 为准。

## 边界与验证

稳定能力必须能由当前公共 API 和仓库案例复现。尚未进入实现的设想会标为 planned 或 research，不能据此假设软件已经支持相应计算。
