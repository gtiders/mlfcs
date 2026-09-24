# K4As4Pt2 FC2–FC4 拟合

本案例将旧 `tutorial_old/fitting/K4As4Pt2` 的联合拟合迁移到当前 API。训练结构中的 ASE
calculator 已保存力，因此脚本直接从 `train.extxyz` 读取 `Atoms` 并组装拟合系统，不需要
重新运行外部计算器。

拟合使用 10 原子原胞和 $2\times2\times3$、120 原子的参考超胞，联合拟合 FC2、FC3、FC4。
截断半径沿用旧案例：FC2 为 6.5 Å，FC3 为 $12a_0$，FC4 为 $8a_0$；最大体阶分别为 2、3、3。
默认解法是列缩放 MINRES，拟合后再对三个阶分别做平移不变性投影。此后输出 MLFCS 原生模型、
Phonopy FC2 文本文件，以及 ShengBTE FC3 和 FC4 文件。

从本目录运行：

```bash
uv run python fit.py
```

每次运行都会覆盖本目录的 `fit.log`，其中包含完整 stdout、stderr 和 traceback；拟合摘要写入
`metrics.json`。原胞、参考超胞和训练数据与 `fit.py` 一起放在本目录。
`train.extxyz` 是由本案例的原胞、超胞及 ALAMODE 随机位移数据转换得到的训练集。

## 热导率

`kappa/` 从 `force_constants.mlfcs` 导出完整稠密 FC2/FC3 HDF5，并使用 phono3py 的 RTA
计算 10×10×10 网格、300–900 K（间隔 100 K）的晶格热导率。导出文件保留 MLFCS 当前的完整
HDF5 格式，不转换为紧凑存储。运行：

```bash
uv run --group reference python kappa/run.py
```

运行日志、输入 FC2/FC3 HDF5、热导率 HDF5 和参数摘要都保存在 `kappa/`。
