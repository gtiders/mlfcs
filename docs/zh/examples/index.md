---
title: 案例
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 案例

案例是可复现的物理与数值证据，不是 Tutorial 的替代品。每个仓库案例独立拥有输入、脚本、来源、预期输出和保留图像；生成缓存和巨大力常数文件保持在本地，除非明确标记为参考数据。

## 有限差分与拟合

- [Si 有限差分](si-finite-difference.md)、[拟合](si-fitting.md)与[输运交接](si-transport.md)
- [K4As4Pt2 有限差分](k4as4pt2-finite-difference.md)、[拟合](k4as4pt2-fitting.md)与[输运](k4as4pt2-transport.md)
- [SnSe 高阶拟合](snse-fitting.md)
- [Ba8Ga16Ge30 温度相关拟合](ba8ga16ge30-md-fitting.md)与[输运](ba8ga16ge30-transport.md)

## 约束与温度相关声子

- [MoS2](mos2-rotational.md)和[石墨烯](graphene-rotational.md)旋转约束
- [K4As4Pt2 SCPH](k4as4pt2-scph.md)与[SSCHA](k4as4pt2-sscha.md)
- [KCl SSCHA](kcl-sscha.md)

## 映射回归

- [非对角超胞回归](non-diagonal-supercell.md)

从仓库根目录使用 `uv run` 执行案例脚本。运行前必须阅读案例 README，因为可选 calculator 和下游应用有意不作为基础依赖。
