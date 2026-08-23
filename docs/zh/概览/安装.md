---
title: 安装
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 安装

MLFCS 要求 Python 3.12 或更高版本。

```bash
uv sync
```

已有环境也可以使用：

```bash
python -m pip install .
```

基础依赖是 ASE、NumPy、SciPy、spglib、h5py 和 JAX。具体 calculator 与后处理软件保持可选；
例如绘制 phonopy/SeeK-path 声子谱时再使用 `uv run --with phonopy --with seekpath`。

正式计算前还应确认 ASE、spglib、JAX 与所选计算器能够在同一环境中正常导入。外部声子或输运程序不属于 MLFCS 的基础运行依赖，应按具体工作流单独安装并记录版本。
