# 数值验证与持续集成

[English](VALIDATION.md) | 中文

## 验证目标

测试需要分别回答三个问题，不能仅用“文件能够写出”代替数值验证：

1. 有限差分、对称性展开与力常数重建是否正确；
2. 原子顺序、晶格平移约定和张量指标是否能与外部程序对应；
3. HDF5、ShengBTE 和 phonopy 等输出是否保持相同的物理数据。

## 独立的 AlN 三阶基准

CI 中的 AlN 基准来自 phono3py 官方 `example/AlN-rd` 数据集。用
pypolymlp 0.20.4 在完整的 200 个结构上一次性训练势函数，然后使用同一个势函数
分别产生：

- MLFCS `sow()` 顺序中所有结构的力；
- phono3py 系统有限位移所需的力以及 traditional solver 得到的完整 FC3。

两条路径都使用 2x2x2 超胞和 0.01 Å 位移，并覆盖该超胞内全部 MIC 原子对。
MLFCS 直接使用 `5.8760168278 Å` 半径，即最大 MIC 距离 `5.8760158278 Å` 加
`1e-6 Å` 数值余量；phono3py 保持默认行为，不设置 `cutoff_pair_distance`。
这是相同有限超胞上的全覆盖比较，不代表无限晶体的截断半径已经收敛。CI 不重新训练
势函数，也不运行 pypolymlp；它只读取压缩后的力和参考 FC3，因此运行时间和内存占用稳定。

比较时关闭 MLFCS ASR，使两边都保持原始有限差分结果，避免把不同的约束投影方法
混入有限差分验证。测试只比较 MLFCS 截断范围覆盖的原子三元组。

## hiphive 的作用

hiphive 仅属于开发依赖和独立验证工具，不参与 MLFCS 的计算实现。测试先完成：

1. 将 MLFCS 的 `(n_primitive, n_supercell, n_supercell, 3, 3, 3)` 平移约化表示
   展开为完整超胞 FC3；
2. 按元素、周期性最小镜像距离匹配 MLFCS 与 phono3py 的原子顺序；
3. 通过 hiphive `ForceConstants.from_arrays` 将双方规范化为同一种完整 FC3 表示；
4. 比较最大绝对误差和 RMS 误差。

这使 hiphive 成为格式与表示的第三方适配器，而不是被验证算法的一部分。

安装 ALAMODE 的 `anphon` 后，下列可选集成测试会让真实的 harmonic reader 读取
MLFCS 写出的最小 FCSXML，以及半超胞边界的双重 27 镜像展开。若其不在 `PATH`，可通过
`MLFCS_ANPHON` 指定：

```bash
MLFCS_ANPHON=/path/to/anphon uv run pytest -q tests/integration/test_alamode_anphon.py
```

当前固定夹具覆盖 32000 个原子三元组：FC3 最大量级约 68.69 eV/Å³，最大绝对差
约 0.01692 eV/Å³，RMS 差约 0.000464 eV/Å³，相对二范数误差约
`2.87e-4`，相关系数约 `0.9999999615`。CI 同时限制这些指标，避免仅靠相关系数
掩盖系统误差。

## 独立的 AlN 二阶基准

同一个 AlN pypolymlp 势函数还用于比较 MLFCS 与 phonopy traditional solver 的
完整 FC2。双方使用相同的 2x2x2 超胞、0.01 Å 位移和全超胞覆盖条件；MLFCS 使用
12 个中心差分构型，phonopy 使用 4 个对称性选择构型，因此验证的是最终 FC2，而不是
要求两边拥有相同的位移计划。

未施加 MLFCS ASR 时，FC2 最大量级约为 `21.4348 eV/Å²`，最大绝对差约为
`0.003326 eV/Å²`，RMS 差约为 `0.000357 eV/Å²`，相对二范数误差约为
`1.48e-4`，相关系数约为 `0.9999999933`。双方分别施加 ASR 后，相对二范数误差
约为 `1.43e-4`，相关系数约为 `0.9999999944`；MLFCS 与 phonopy 的 ASR 残差
分别约为 `5.50e-14` 和 `1.08e-14 eV/Å²`。

## 有 ASR 的交叉验证

第二项测试显式比较 MLFCS `acoustic_sum_rule=True` 与 phono3py traditional
`symmetrize_fc3(level=3)` 的结果。原始 phono3py FC3 的最大平移求和残差约为
`0.01162 eV/Å³`；投影后 MLFCS 与 phono3py 的残差分别约为 `2.58e-13` 和
`5.53e-14 eV/Å³`，双方都严格满足 ASR。

投影后共同支持上的最大绝对差约为 `0.01679 eV/Å³`，RMS 差约为
`0.000851 eV/Å³`，相对二范数误差约为 `0.0527%`，相关系数约为
`0.9999998627`。全超胞覆盖消除了此前由双方约束空间支持集不同造成的大部分投影差异。

## hiPhive BaGaGe 复杂材料公开基准

可选的复杂材料基准使用 hiPhive 公开的 Ba8Ga16Ge30 笼状化合物 200 快照数据：100 个
Monte-Carlo rattle 快照，以及 300 K 与 650 K 各 50 个 MD 快照。两条路径严格采用论文中
的双体 FC2+FC3+FC4 模型、`[5.40, 4.35, 4.35] Å` 截断、R3 的 54 原子晶胞和平移 ASR。
共同物理参数空间为 25,495 维，精确 ASR 零空间为 6,052 个拟合坐标。

在全部 200 个结构上拟合（这里是训练拟合，而不是论文的 10 折交叉验证）时，hiPhive 的力
RMSE 为 `49.10 meV/Å`，MLFCS 为 `57.60 meV/Å`。这不是错误：hiPhive 拟合普通 Taylor
特征，MLFCS 拟合按协方差正交化的 Wick 特征后再转换为 Taylor IFC。显式对齐原子和张量轴
后，FC2/FC3/FC4 的相对 RMS 差分别为 `1.62%`、`18.63%`、`47.64%`，且所有簇均成功匹配。
因此该结果是严格的**方法对照**，不能表述为字节或张量逐项相等。

同一数据上的 8,192 个随机特征还说明 Wick 的作用是缓解而非消除污染：Taylor/Wick 的
线性-三次特征相关均值为 `0.06409/0.06280`，RMS 为 `0.08094/0.07867`，最大值从
`0.68239` 降至 `0.40341`。

2026-08-15 还完成了更严格的检查：对同一个 FC2+FC3+FC4 模型流式累加**完整**的 ASR
约化物理设计，只比较 FC2--FC4 交叉 Gram 块。它包含求解器实际使用的对称性基、双体支撑、
ASR 零空间及列归一化。Wick 将最大逐列归一化相关从 `0.51662` 降至 `0.21352`，RMS 从
`0.01683` 降至 `0.00768`；归一化联合 Gram 条件数从 `2.63e6` 降至 `1.37e6`。但最大
**子空间** canonical correlation 从 `0.94515` 变为 `0.96235`。因此可确认 Wick 在此例中
降低了直接的列级 FC2--FC4 耦合，却不能保证两个完整约束子空间必然更正交。

资源数据均在开发机上用 `/usr/bin/time -v` 串行采集。MLFCS 在 Gram 前进行 ASR 零空间
参数化，约化 Gram 仅 279.4 MiB；成功的缓存恢复求解为 64.78 s、峰值 RSS 1.46 GiB。此前
冷启动的 Gram 构造为 65.96 s，并在默认 1,000 次迭代上限不足时达到 2.01 GiB；上限提高到
10,000 后严格收敛。hiPhive 基线在显式物化设计矩阵时观测到约 13 分钟墙钟和 6.48 GiB
峰值 RSS。它们是本机测量，不应当作为跨机器的绝对性能承诺。复现时应严格逐进程运行：

Gram 恢复缓存也在空临时目录中独立测量：200 快照 BaGaGe 的冷 Gram 为 `67.99 s`，紧接着
验证过的热缓存命中为 `0.0689 s`，约 `987x`。端到端命令仍需 115.70 s，因为每次都会重新
构建对称性/ASR 参数化和 JAX 静态程序；与此前 65.96 s 的冷 Gram 相比，67.99 s 约 3% 的
差异属于测量波动或缓存写入开销，不表示缓存使核心 Gram 变慢。

```bash
/usr/bin/time -v uv run python reference_tools/benchmark_hiphive_examples.py bagage-hiphive --validation-split 0.0
/usr/bin/time -v uv run python reference_tools/benchmark_hiphive_examples.py bagage-mlfcs --validation-split 0.0
uv run python reference_tools/benchmark_hiphive_examples.py bagage-compare
uv run python reference_tools/benchmark_hiphive_examples.py bagage-wick
uv run python reference_tools/benchmark_hiphive_examples.py bagage-collinearity
/usr/bin/time -v uv run python reference_tools/benchmark_hiphive_examples.py bagage-gram-cache
```

## 原生 SSCHA 参考

解析谐振模型分别检查经典协方差、量子零点位移、虚频策略、可选位移裁剪以及
由采样力恢复 FC2 的数值正确性。另一项仅在开发环境运行的 phonopy 参考比较
相容 q 点频率和量子采样协方差。phonopy 只用于该独立参考，不参与 SSCHA
实现，也不是基础运行时依赖。

端到端 KCl 参考进一步使用 phonopy 官方的 120 结构 pypolymlp 势函数、8 原子常规胞、
2x2x2 超胞、300 K 和种子 42。串行 CI 版本使用 10 个快照和 1 轮 canonical 迭代；
K 原子自相互作用块约为 `2.1625 eV/Angstrom^2`，位于 phonopy 官方
`2.1 +/- 0.1` 的验收范围内。归一化至每个原胞的自由能约为 `-0.0949 eV`，
与 phonopy 稠密网格、三轮迭代参考值相差约 `3.7 meV`。该差异包含有限采样噪声，
以及相容 q 点与稠密网格谐振自由能约定的明确差别。

## CI 分层

GitHub Actions 分为三个相互独立的任务：

- `unit-and-api`：在 Python 3.12 和 3.13 上运行 Ruff、格式检查以及所有非参考测试；
- `scientific-reference`：在 Python 3.12 上依次独立运行 hiphive 适配器、训练材料
  SHA-256 溯源、AlN FC2、无 ASR 和有 ASR 的 AlN FC3，以及谐振采样对比；
- `package`：构建 Python sdist 和 wheel。

BLAS、OpenMP 和 JAX CPU 后端均限制为单线程，避免小型 CI 任务因嵌套并行产生不稳定
内存峰值。官方 AlN 势函数的重新训练是维护者基准，不属于每次 push 的 CI。

2026-08-15 的完整本地运行共收集并通过 119 项测试，墙钟 321.41 s、峰值 RSS 2.56 GiB；
Ruff 成功通过，耗时 0.03 s、峰值 RSS 35.5 MiB。AlN FC3 夹具迁移本身为 11.35 s、375 MiB，
迁移后的 AlN FC3 参考测试为 7.34 s、350 MiB。

完整目录约定和执行命令见 `tests/README.md`。

## 数据来源和再生成

夹具的固定上游提交、许可证、软件版本与生成命令记录在
`tests/reference/phono3py/AlN_FC3/data/README.md`。生成程序为
`reference_tools/generate_AlN_phono3py_fixture.py`。

同一目录的 `data/training/` 还保存 phono3py 官方 200 结构 AlN 训练数据和本次实际
使用的 `polymlp.yaml`。普通 CI 只读取 `reference.npz`，维护者可以完全使用仓库内
文件重新生成该夹具，无需再次下载或训练势函数。
