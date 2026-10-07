# 教程总览

本站的教程以 Jupyter notebook 形式维护：文档即教程，教程即文档。每个 notebook
既讲解一段工作流，又包含**真实执行过的完整输出**——图、拟合指标与收敛历史都由
notebook 在本仓库环境中运行生成，并随源码一起提交。改代码、改文字、改数据都
发生在同一个地方。

## 教程列表

| Notebook | 内容 | 涉及的 API |
|---|---|---|
| [石墨烯与 MoS₂：转动求和规则](rotational-sum-rules.ipynb) | 同一 FC2 拟合在 ASR 与 ASR + Born–Huang + Huang 两种投影下的声子谱对比 | `ClusterSpace`、`ClusterMap`、`FitSystem`、`enforce_asr`、`enforce_rotation` |
| [Si：有限差分与外推](si-finite-difference.ipynb) | 用 ASE Tersoff 势做多步位移的 FC2/FC3 有限差分与外推，并计算声子谱与热导率 | `FiniteDifference`、`enforce_asr`、导出 API |
| [Ba8Ga16Ge30：FC2+FC3 拟合与声子谱](ba8ga16ge30.ipynb) | 54 原子笼形化合物从 300 K NVE 快照拟合有效 FC2+FC3，再画谐波声子谱 | `FitSystem`、`enforce_asr`、`Harmonic` |
| [Si：FC2–FC5 联合拟合与热导率](si-fitting.ipynb) | 四阶联合拟合、壳截断语义、`EstimateCutoff` 工具与 phono3py 热导率 | `FitSystem`、`EstimateCutoff`、`rank_info` |
| [K4As4Pt2：FC2–FC4 拟合、热导率与 SCPH](k4as4pt2.ipynb) | FC2–FC4 联合拟合、RTA 热导率与温度相关的 SCPH 声子谱 | `FitSystem`、`SCPH`、`Harmonic` |
| [NaCl：短程拟合与 NAC 声子谱](nacl-long-range.ipynb) | 长程力扣除、短程拟合、总 FC2 导出，再用 Phonopy NAC 比较声子谱 | `ForceDataset`、`DipoleEwald`、`CompactForceConstants`、`FitSystem` |

API 各页（[有限差分](../finite-difference-api.md)、[拟合](../fitting-api.md)、
[谐波声子](../harmonic-api.md)、[SCPH](../scph.md)）给出精确的接口契约；教程
负责把这些接口串成完整工作流。整数核与开发者参考见
[整数核与开发者参考](../numba-integer-audit.md)一节。

## 运行环境

教程依赖 `tutorial` 依赖组（phonopy、phono3py、matplotlib 以及
notebook 执行工具）：

```bash
uv sync --group docs --group tutorial
```

逐个执行 notebook（保持串行，一次只跑一个，部分拟合任务的内存峰值较高）：

```bash
uv run --group docs --group tutorial jupyter nbconvert --to notebook \
    --execute --inplace docs/notebooks/<name>.ipynb
```

## 数据与约定

- 训练数据与结构文件在 `docs/notebooks/data/` 下，按案例分目录，随仓库提交，
  因此所有 notebook 都可以直接重跑。
- notebook 演示中的导出文件（`.mlfcs`、HDF5、FORCE_CONSTANTS 文本）写入临时
  目录，不污染文档树；正式工作流中把 `SCRATCH` 换成你的工作目录即可。
- 数据来源与归属：Si 训练集与截断壳语义来自远程 SI 布局；Ba8Ga16Ge30 的
  300 K 快照取自 hiPhive 的笼形热导率示例（沿用其归属与许可条款）；
  K4As4Pt2 的位移数据由 ALAMODE 生成。

整数核的安全性契约（教学案例为什么可以放心跑大体系）见
[本地整数契约与清理规则](../local-integer-contracts.md)。
