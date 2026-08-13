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

完整目录约定和执行命令见 `tests/README.md`。

## 数据来源和再生成

夹具的固定上游提交、许可证、软件版本与生成命令记录在
`tests/reference/phono3py/AlN_FC3/data/README.md`。生成程序为
`reference_tools/generate_AlN_phono3py_fixture.py`。

同一目录的 `data/training/` 还保存 phono3py 官方 200 结构 AlN 训练数据和本次实际
使用的 `polymlp.yaml`。普通 CI 只读取 `reference.npz`，维护者可以完全使用仓库内
文件重新生成该夹具，无需再次下载或训练势函数。
