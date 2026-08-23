---
title: MLFCS 是什么
audience:
  - beginner
status: stable
code_verified: 4.0.0a4
---

# MLFCS 是什么

## 定位与范围

本页把“MLFCS 是什么”放回 MLFCS 的完整工作流中，说明它解决什么问题、与哪些模块相连，以及哪些工作明确不由它承担。MLFCS 是一个以 ASE 为公共边界、从原子力构造对称约化谐性与非谐力常数的 Python 库。它专注于力常数计算、拟合、约束、超胞展开与导出，而不是替代下游声子或输运求解器。

## 如何使用本页

先确认输入结构、单位和目标后处理程序，再沿页面给出的入口进入 Concepts、Tutorials 或 How-to。不要把概览页当作参数参考；可执行签名以 Reference 为准。

## 边界与验证

稳定能力必须能由当前公共 API 和仓库案例复现。尚未进入实现的设想会标为 planned 或 research，不能据此假设软件已经支持相应计算。
