# K4As4Pt2 FC2–FC4 拟合

本案例将旧 `tutorial_old/fitting/K4As4Pt2` 的联合拟合迁移到当前 API。训练结构中的 ASE
calculator 已保存力，因此脚本直接从 `train.extxyz` 读取 `Atoms` 并组装拟合系统，不需要
重新运行外部计算器。

拟合使用 10 原子原胞和 $2\times2\times3$、120 原子的参考超胞，联合拟合 FC2、FC3、FC4。
截断半径沿用旧案例：FC2 为 6.5 Å，FC3 为 $12a_0$，FC4 为 $8a_0$；最大体阶分别为 2、3、3。
默认解法是列缩放 MINRES，拟合后再对三个阶分别做平移不变性投影。此后输出 `fc-fit.mlfcs`、
`fc2-phonopy.txt`，以及 `fc3-shengbte.txt` 和 `fc4-shengbte.txt`。

从本目录运行：

```bash
uv run python fit.py
```

每次运行都会覆盖本目录的 `fit.log`，其中包含完整 stdout、stderr、traceback、拟合摘要和
ASR 投影诊断。原胞、参考超胞和训练数据与 `fit.py` 一起放在本目录。
`train.extxyz` 是由本案例的原胞、超胞及 ALAMODE 随机位移数据转换得到的训练集。

## 热导率

`kappa/` 从 `fc-fit.mlfcs` 导出完整稠密 FC2/FC3 HDF5，并使用 phono3py 的 RTA
计算 10×10×10 网格、300–900 K（间隔 100 K）的晶格热导率。导出文件保留 MLFCS 当前的完整
HDF5 格式，不转换为紧凑存储。运行：

```bash
uv run --group reference python kappa/run.py
```

运行日志、输入 FC2/FC3 HDF5、热导率 HDF5 和参数摘要都保存在 `kappa/`。

## 声子谱对比

`phonon/compare.py` 使用 seekpath 构造高对称路径，分别读取本目录的原生
`fc-fit.mlfcs` 和已导出的 Phonopy `fc2-phonopy.txt`，计算并绘制
两套 FC2 的声子谱。脚本会核对 Phonopy 超胞与导出 FC2 所用超胞的原子顺序。

```bash
uv run --group reference --with seekpath --with matplotlib python phonon/compare.py
```

数值差异、单位换算因子及最差 q 点写入 `phonon/comparison.json`；叠加的声子谱和
逐 q 点最大绝对差绘于 `phonon/band-comparison.png`。

## 自洽声子谱

`scph/run.py` 使用拟合所得 FC2 与 FC4 进行静态四阶 loop SCPH。倒空间网格为整数
$4\times4\times6$，只在不可约代表点对角化；0–900 K 每隔 100 K 计算一次，
从最高温向低温求解，仅在前一温度收敛时用其重整化 FC2 作为下一温度的初值。
结果仍按温度升序返回。收敛判据是星权重 RMS 频率变化小于
$10^{-9}$ THz；脚本会拒绝任何未达到该判据的温度点。

```bash
uv run --with seekpath --with matplotlib python scph/run.py
```

每个温度的迭代历史、高对称路径频率保存在 `scph/results.json`，迭代摘要在
`scph/run.log`，叠加声子谱在 `scph/temperature-bands.png`。Γ 点声学支约
$10^{-6}$ THz 的带符号残差是数值 ASR 漂移，未作为虚频容忍或裁剪阈值。
迭代中若试探动力学矩阵有负曲率，协方差使用该曲率的绝对值，输出频率仍保留符号；
这是试探协方差的显式延拓规则，并不把虚频判定为计算失败。另报告排除 Γ 点
平移模后的最小频率与虚频状态；它们是物理诊断，不与数值迭代收敛混为一谈。
