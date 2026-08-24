---
title: 安装与快速上手
audience:
  - beginner
status: stable
code_verified: 4.0.0a6
---

# 安装与快速上手

本页介绍如何安装 MLFCS，并用最短的路径完成一次力常数计算。

## 环境要求

- Python 3.11+
- 推荐使用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境与依赖。

## 安装

### 从源码安装（开发方式）

```bash
git clone https://github.com/cloudac7/mlfcs.git
cd mlfcs
uv sync
```

### 作为依赖安装

```bash
uv add mlfcs
# 或
pip install mlfcs
```

## 最短上手：有限差分计算 FC2

```python
from mlfcs import FiniteDifferenceCalculation

# TODO: 补充一个可运行的最小示例（结构准备 -> 计算配置 -> 运行 -> 读取结果）
```

## 下一步

- 阅读[教程总览](tutorials/index.md)选择一个完整工作流。
- 查看 [API 参考](reference/index.md)了解各顶层导出的用法。
- 有疑问请看[问答](Q%26A.md)。
