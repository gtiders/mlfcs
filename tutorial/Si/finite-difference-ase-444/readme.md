# Si 4x4x4 finite-difference and phono3py workflow

本案例使用 2 原子 Si primitive、$4\times4\times4$ reference supercell（128 原子）和 `Si_2022_NEP3_5body.txt` NEP 势。

## 有限差分

```bash
uv run python run.py
```

`run.py` 使用中心有限差分分别计算 FC2 和 FC3：

- FC2：6 个 NEP 力评估；
- FC3：132 个 NEP 力评估；
- 位移步长：$0.01$ Å；
- ASR：开启；
- FC3 截断半径：$7.7237404951$ Å；
- FC2 输出：`force_constants.hdf5`；
- FC3 输出：`fc3.hdf5`，phono3py HDF5 格式。

## phono3py 热导率

```bash
uv run python thermal_conductivity.py
```

脚本一次向 phono3py 传入全部温度，避免逐温度调用覆盖同一个输出文件：

- 温度：300、400、500、600、700、800、900、1000、1100 K；
- 网格：$15\times15\times15$；
- 同位素散射：开启；
- 输出：`kappa-m151515.m151515.hdf5`；
- 结果索引：`thermal-conductivity.json`。

本次计算得到的对角热导率（W/m-K）为：

| 温度 (K) | $\kappa_{xx}$ | $\kappa_{yy}$ | $\kappa_{zz}$ |
|---:|---:|---:|---:|
| 300 | 91.668 | 91.668 | 91.668 |
| 400 | 66.058 | 66.058 | 66.058 |
| 500 | 51.970 | 51.970 | 51.970 |
| 600 | 42.959 | 42.959 | 42.959 |
| 700 | 36.665 | 36.665 | 36.665 |
| 800 | 32.005 | 32.005 | 32.005 |
| 900 | 28.410 | 28.410 | 28.410 |
| 1000 | 25.548 | 25.548 | 25.548 |
| 1100 | 23.215 | 23.215 | 23.215 |

`run.log` 和 `thermal-conductivity.log` 保存了完整运行日志。

## FC3 五步长外推

```bash
uv run python run_fc3_extrapolation.py
uv run python thermal_conductivity_fc3_extrapolated.py
```

FC3 外推使用 $h=0.010,0.015,0.020,0.025,0.030$ Å 的中心差分结果，并对 $h^2$ 做二次零步长外推。其配置和诊断为：

- 660 个 NEP 力评估；
- FC3 截断半径：$7.7237404951$ Å；
- 中心步长结果到零步长的相对 L2 修正：$6.5754483570\times10^{-4}$；
- 最大多项式残差：$1.6248560719\times10^{-8}$ eV/Å$^3$；
- 输出 FC3：`fc3-extrapolated.hdf5`；
- 热导率输出：`kappa-m151515.m151515-extrapolated-fc3.hdf5`。

外推 FC3 与原 FC2 的对角热导率（W/m-K）为：

| 温度 (K) | $\kappa_{xx}$ | $\kappa_{yy}$ | $\kappa_{zz}$ |
|---:|---:|---:|---:|
| 300 | 91.720 | 91.720 | 91.720 |
| 400 | 66.095 | 66.095 | 66.095 |
| 500 | 52.000 | 52.000 | 52.000 |
| 600 | 42.984 | 42.984 | 42.984 |
| 700 | 36.686 | 36.686 | 36.686 |
| 800 | 32.024 | 32.024 | 32.024 |
| 900 | 28.426 | 28.426 | 28.426 |
| 1000 | 25.563 | 25.563 | 25.563 |
| 1100 | 23.228 | 23.228 | 23.228 |

完整有限差分日志保存在 `fc3-extrapolation.log`，热导率日志保存在 `thermal-conductivity-extrapolated-fc3.log`。