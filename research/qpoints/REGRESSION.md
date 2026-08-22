# q 点映射与非谐案例回归

## 覆盖范围

本轮只验证当前 HNF reciprocal quotient 和使用它的 SSCHA/SCPH 路径，不修改 q 点算法。

自动测试覆盖：

- 对角整数超胞的规则网格；
- 非对角剪切矩阵；
- HNF canonical ordering；
- HNF 方法与旧 residue 搜索得到相同 q 点集合；
- q 点数量和唯一性；
- reciprocal commensurability；
- SSCHA 的 $q$ 与 $-q$ 配对；
- SCPH 单 worker 与多 worker 的 covariance 一致性。

测试命令：

```bash
uv run pytest -q -s tests/test_anharmonic_qpoints.py tests/test_sscha_ensemble.py tests/test_anharmonic_scph.py
```

全部通过。

## K4As4Pt2 SCPH

参数保持案例定义：

- 温度为 300、600、900 K；
- `interpolation_multiplier=1`；
- `scph_multiplier=2`；
- `mixing=0.5`；
- tolerance 为 $10^{-10}$ THz；
- 最多 200 次迭代；
- 4 个 q 点 worker。

三个温度均运行满 200 步，没有满足严格阈值。最后一步频率变化为：

| 温度 | 最后变化量 | 最低频率 | 最高频率 |
|---:|---:|---:|---:|
| 300 K | $3.99\times10^{-6}$ THz | $-9.26\times10^{-5}$ THz | 7.9434 THz |
| 600 K | $6.19\times10^{-7}$ THz | $-1.14\times10^{-4}$ THz | 7.8083 THz |
| 900 K | $1.38\times10^{-6}$ THz | $-1.22\times10^{-4}$ THz | 7.6873 THz |

全部频率数组均为有限值。末期变化停留在 $10^{-6}$ THz 左右的数值平台，因此本轮不把 `converged=False` 隐藏为成功，也不修改 tolerance。

声子谱：`examples/scph/K4As4Pt2/results/phonopy-seekpath-harmonic-vs-scph.png`。

## K4As4Pt2 SSCHA

参数保持案例定义：300 K、每轮 100 个快照、5 次更新、随机种子 42、直接更新 mixing 1.0。采样 reference 含 12 个 q 点，所有 357 个模式均被采样，没有虚频模式或被排除模式。

最后一次更新的相对 FC2 变化为 $1.996\times10^{-2}$。生成的 FC2 数组全部为有限值。

声子谱：`examples/sscha/K4As4Pt2/figures/harmonic_vs_sscha.png`。

## 结论

当前 q 点 quotient、非对角矩阵约定、$q/-q$ 配对和并行 SCPH 路径没有发现回归。SCPH 未达到 $10^{-10}$ THz 是固定点迭代停在数值噪声平台的问题，不是 q 点集合缺失、重复或不相容。后续固定点加速应作为独立阶段处理。

## KCl SSCHA 对照

KCl 案例也已按原始配置完整重跑：600 K、每轮 100 个快照、50 次迭代、随机种子 42。phonopy 和 MLFCS 两条路径均完成，随后重新生成声子谱和自由能图。

绘图时修复了案例层的格式适配：MLFCS 保存的 NumPy 数组是相对于用户 8 原子 primitive 的 compact FC2，而 phonopy YAML 自动选择了 2 原子 primitive；绘图现在读取 MLFCS 已导出的 64×64 phonopy 文本 FC2，不再错误地尝试把 compact 第一轴当成 phonopy primitive site。

重跑结果暴露出尚未解释的实质差异：MLFCS 与 phonopy 的最终有效谐波声子谱明显不同，自由能平台也分别约为 $-0.199$ eV/atom 和 $-0.205$ eV/atom。两者不是同一随机采样实现，不能要求逐轮一致，但当前差异大到不能作为已通过的物理回归。需要单独核查采样 covariance、质量加权、自由能归一化、FC2 更新规则以及两套方法实际使用的 primitive/reference 定义。

旧结果比较脚本无法运行，因为它仍指向已经不存在的 `examples/cases/KCl/sscha/output`。本轮没有伪造或更新 legacy baseline。
