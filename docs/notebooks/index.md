# 教程总览

这些案例从真实材料的结构和原子力出发，把模型、超胞和数据集串成可执行的工作流。每份 notebook 都保存了从干净 kernel 顺序执行的输出，包括力误差、图和结果表。

如果第一次使用 MLFCS，先读 [核心概念](../core-concepts.md)，了解应该把参考结构、超胞和带力样本分别交给哪个对象。随后按数据来源选择案例：已有力文件可以直接拟合，需要自己生成位移则从 Si 有限差分开始。

## 案例与阅读顺序

| Notebook | 最后得到什么 |
|---|---|
| [NaCl：长程力与 NAC](nacl-long-range.ipynb) | 扣除长程力、拟合短程 FC2、回加后用 Phonopy 比较 NAC 声子谱 |
| [石墨烯与 MoS₂：转动求和规则](rotational-sum-rules.ipynb) | 每种材料拟合一次 FC2，比较旋转修正前后的误差与声子谱 |
| [Si：有限差分与外推](si-finite-difference.ipynb) | 生成位移、用 Tersoff 计算力、恢复 FC2 和 FC3，再检查声子与 300 K RTA 热导率 |
| [Ba8Ga16Ge30：轨迹拟合](ba8ga16ge30.ipynb) | 从 300 K 分子动力学快照联合拟合 FC2、FC3，并画有效二阶模型的声子谱 |
| [Si：FC2–FC5 联合拟合](si-fitting.ipynb) | 用一套带力样本联合恢复多阶力常数，再计算固定模型的 RTA 热导率 |
| [K4As4Pt2：热导率与 SCPH](k4as4pt2.ipynb) | 拟合 FC2–FC4，计算 RTA 热导率，并比较四个温度的 SCPH 有效声子谱 |

## 运行准备

在仓库根目录安装教程与文档依赖：

```bash
uv sync --group docs --group tutorial
```

然后打开 `docs/notebooks/` 下的 notebook，选择对应环境的 Python kernel，从第一格依次运行。数据路径兼容 notebook 目录与仓库根目录。全部输入文件随仓库提供，不需要另行下载或手工生成；需要外部力的替换方法在各案例中说明。

也可以从仓库根目录逐份执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4 \
uv run --group docs --group tutorial jupyter nbconvert --to notebook \
    --execute --inplace --ExecutePreprocessor.timeout=7200 \
    docs/notebooks/nacl-long-range.ipynb
```

更换命令末尾的文件名即可运行其他案例。保持串行，一次只运行一个 notebook；大模型的初始化、拟合与外部 FC3 数组都可能占用较多内存。命令中将 BLAS 与 OpenMP 限制为单线程，Numba 使用 4 线程，控制线程竞争并保留力方程构造的并行能力。这是运行资源设置，不改变模型定义；按自己的机器调整线程数时，仍应保持 notebook 串行。

## 数据与结果

- 数据目录是 `docs/notebooks/data/`，按材料分开。Si 使用配套 Tersoff 势与已有训练样本；Ba8Ga16Ge30 轨迹沿用 hiPhive 笼形材料示例；K4As4Pt2 使用 ALAMODE 位移样本。NaCl 的来源与许可证见仓库中的 `docs/notebooks/data/nacl/README.md`。重用数据时保留原有归属与许可。
- 预计算力由 `ForceDataset` 读取，MLFCS 不会隐式调用 calculator。Si 有限差分案例则明确使用 ASE Tersoff 计算并保存每帧力。
- 中间 HDF5 文件写入临时目录，完整结果保存在 notebook 输出中。正式研究可将 `SCRATCH` 换成持久工作目录；保存参数模型使用 `ForceConstants.save()`。
- 教学截断、倒空间网格与误差表不等于材料性质已经收敛。每份教程说明了换材料时应修改的输入和继续检查的方向。

接口选项见 [有限差分](../finite-difference-api.md)、[拟合](../fitting-api.md)、[长程力](../ewald-api.md)、[谐波声子](../harmonic-api.md)与 [SCPH](../scph.md)。数学背景见 [Theory](../theory/index.md)，资源范围见[数值契约](../development/numerical-contracts.md)。
