# MLFCS

[![文档构建](https://github.com/gtiders/mlfcs/actions/workflows/ci.yml/badge.svg)](https://github.com/gtiders/mlfcs/actions/workflows/ci.yml)
[![文档站](https://img.shields.io/badge/docs-GitHub%20Pages-0f766e)](https://gtiders.github.io/mlfcs/zh/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776ab)](https://www.python.org/)
[![许可证：GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)

[English](README.md) | 简体中文

<!-- BEGIN GENERATED: docs/zh/index.md -->

<p align="center">
  <img src="https://gtiders.github.io/mlfcs/assets/images/logo.png" alt="MLFCS" width="560">
</p>



MLFCS 是一个以 ASE 为公共边界、从原子力构造对称约化谐性与非谐力常数的 Python 库。它提供有限差分、仅力数据拟合、物理约束、稀疏 primitive 实空间存储、温度相关有效谐波工作流，以及面向下游声子和输运软件的显式导出。

## 为什么需要 MLFCS

高阶力常数同时面对快速增长的相互作用空间、Cartesian 张量对称性、大规模训练数据，以及不同 Taylor 阶之间相互耦合的误差。MLFCS 将结构、对称约化 interaction、多项式基、拟合、力常数和格式转换保持为可检查的显式阶段，使每项近似都能够独立验证。

## 核心能力

- FC2 到任意已支持高阶的有限差分，主要生产验证集中在 FC2–FC4。
- 在一个固定 reference supercell 上，使用 Taylor 坐标进行仅力数据联合拟合。
- 使用 primitive site 与精确整数平移表示力常数，并以原生 HDF5 v3 稀疏存储。
- 平移约束，以及显式的 FC2 Born–Huang/Huang 后处理。
- 目标超胞 realization，以及 phonopy、phono3py、ShengBTE 和 ALAMODE writer。
- 从 canonical Taylor IFC 构造 ASE Calculator，计算固定晶格上的相对能量与原子力。
- FC4 loop SCPH 和随机有效谐波 SSCHA 工作流。

MLFCS 是 Python 函数库，不是命令行应用。力的产生始终由用户自己的 ASE calculator 或外部电子结构工作流控制。

## 快速开始

安装下载与最短可运行流程见[入门指南](docs/zh/getting-started.md)。

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from mlfcs import FiniteDifferenceCalculation, build_supercell, write_force_constants

primitive = bulk("Al", "fcc", a=4.05)
reference = build_supercell(primitive, (2, 2, 2))
calculation = FiniteDifferenceCalculation(
    primitive,
    reference=reference,
    order=2,
    cutoff=None,
)
fc2 = calculation.run(EMT())
write_force_constants(fc2, "mlfcs.h5", format="hdf5")
```

在把流程用于外部 calculator 或更高阶计算前，先阅读[第一个有限差分 FC2 教程](docs/zh/tutorials/first-fc2-finite-difference.md)。

## 典型工作流

| 目标 | 从这里开始 |
|---|---|
| 使用 ASE calculator 计算力常数 | [有限差分工作流](docs/zh/tutorials/finite-difference-workflow.md) |
| 将位移结构交给 VASP 或其他外部程序 | [外部计算器教程](docs/zh/tutorials/external-calculator.md) |
| 从位移或 MD 快照拟合力常数 | [第一次拟合教程](docs/zh/tutorials/first-fc2-fitting.md) |
| 联合拟合 FC2、FC3 和 FC4 | [联合高阶拟合](docs/zh/tutorials/joint-fc2-fc3-fc4.md) |
| 对 FC2 施加旋转条件 | [旋转约束](docs/zh/tutorials/rotational-constraints.md) |
| 计算 FC4 loop 修正 | [SCPH 工作流](docs/zh/tutorials/scph-workflow.md) |
| 计算随机有效 FC2 | [SSCHA 工作流](docs/zh/tutorials/sscha-workflow.md) |
| 将拟合后的 Taylor IFC 用作 ASE 势 | [ASE Calculator API](docs/zh/reference/calculator-api.md) |

## 计算前先确定结构

在生成力常数前，先确定结果由哪个下游软件使用。优先采用该软件确定的 primitive 和 reference supercell，在力收集全过程中保持 reference 原子顺序不变，并且只对经过验证的整数超胞表示使用 target realization。MLFCS 不会静默重定义 primitive、放大训练超胞、施加应变或执行整体 Cartesian 刚性旋转。

## 当前范围与限制

- 每次拟合只使用一个固定 reference supercell；不支持不同超胞数据联合拟合。
- 三阶和四阶是主要生产验证的高阶路径；更高阶复用稀疏实现，但成本可能无法承受。
- 原生 HDF5 使用 v3 schema；旧原生 schema 被明确拒绝。
- ShengBTE writer 支持 FC3 和 FC4；ALAMODE writer 支持当前实现的 FC2–FC4 映射。
- 尚未实现长程静电力扣除、多极修正和显式 FC3 bubble 自能。
- SCPH 和 SSCHA 必须显式检查收敛性。

[路线图](docs/zh/roadmap/index.md)明确区分稳定、实验、计划、研究与 No-Go 工作。

## 文档地图

### 入门

从[入门指南](docs/zh/getting-started.md)完成安装与最短上手流程。

### 理论

从[理论](docs/zh/theory/index.md)开始阅读完整推导和数值约定。

### 教程

按照[教程学习路线](docs/zh/tutorials/index.md)运行完整、可执行的工作流。

### API

查询[API 参考](docs/zh/reference/index.md)了解每个公开接口的完整签名、参数与语义。

### 参考

[参考文档](docs/zh/reference/index.md)记录公共接口契约、单位约定、诊断与异常。

### 问答

[问答](docs/zh/Q&A.md)按主题归档常见问题与错误信息。

### 路线图

[路线图](docs/zh/roadmap/index.md)区分稳定、计划、研究与 No-Go 工作。

<!-- END GENERATED: docs/zh/index.md -->

## 文档

阅读[中文文档](https://gtiders.github.io/mlfcs/zh/)或[英文文档](https://gtiders.github.io/mlfcs/en/)。源文件分别位于 [docs/zh](docs/zh/) 和 [docs/en](docs/en/)。

## 引用

若 MLFCS 对发表工作有所贡献，请使用 [CITATION.cff](CITATION.cff) 中的软件引用信息。

## 贡献

参见 [CONTRIBUTING_ZH.md](CONTRIBUTING_ZH.md)、[问题追踪](https://github.com/gtiders/mlfcs/issues)和[开发者文档](https://gtiders.github.io/mlfcs/zh/development/)。

## 许可证

MLFCS 按照 [GNU 通用公共许可证第 3 版或更高版本](LICENSE)发布。
