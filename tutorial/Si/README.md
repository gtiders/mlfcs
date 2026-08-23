# Si 二阶力常数有限差分教程

本目录使用 `source/POSCAR.vasp` 和 `source/Si_2022_NEP3_5body.txt`，通过 `calorine` 的 `CPUNEP` 计算 Si 的原子力，再使用 MLFCS 重建 FC2。

## 直接运行 NEP calculator

从仓库根目录执行：

```bash
uv run python tutorial/Si/run_finite_difference.py
```

结果写入 `results/finite-difference/`，包括 MLFCS 原生 HDF5、phonopy 文本格式和 phonopy HDF5 格式。

## 外部 calculator 工作流

先生成位移结构和 manifest：

```bash
uv run python tutorial/Si/prepare_external.py
```

将外部程序的力保存为 `tutorial/Si/work/external/forces/forces-00000.npy` 这类文件。每个文件的形状必须是 `(256, 3)`，并保持 `POSCAR-xxxxx` 对应的原子顺序。然后执行：

```bash
uv run python tutorial/Si/reconstruct_external.py
```

外部流程不会伪造或覆盖力文件；缺少任何一个力文件时，重建脚本会直接失败。
