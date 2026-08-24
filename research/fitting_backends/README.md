# Taylor/Wick 拟合后端对比

## 目的

本研究验证 Taylor 与 Wick 独立拟合后端是否共享正确的约束、求解和物理工作流，并量化两种基底在真实有限训练数据上的差异。它不是性能基准，也不把两个基底的数值相等作为正确性条件。

比较固定使用相同的 primitive、reference supercell、训练快照、interaction cutoff、body order、ASR 和求解参数。两种后端最终都输出 Taylor `ForceConstants`。

## ASR 在哪里施加

拟合流程为：

```text
interaction/orbit 参数
→ 构造逐阶 ASR 矩阵 C
→ 在 null(C) 中求解拟合参数
→ backend lowering
→ Taylor ForceConstants
```

无正则路径显式构造 null-space parameterization；scaled group LASSO 使用隐式 null-space 投影。Taylor lowering 是恒等映射。Wick lowering 会用 reference covariance 收缩成对指标，但该收缩与未收缩 site 指标上的平移求和可交换。因此，若每个 Wick 阶满足 ASR，则 lowering 后的每个 Taylor 阶仍满足 ASR：

$$
C\theta_{\mathrm W}=0
\quad\Longrightarrow\quad
C L\theta_{\mathrm W}=0,
$$

其中 $L$ 是 Wick 到 Taylor 的 lowering。ASR 不需要读取 covariance，也不需要按 backend 实现两个分支。

Born–Huang 和 Huang 旋转条件位于拟合之后，只接收 Taylor FC2。它们同样不依赖拟合基底。

## 真实案例结果

测试日期为 2026-08-24，使用 CPU JAX、FP64 和当前 `dev` 实现。

| 案例 | 阶数 | Taylor 训练 RMSE | Wick 训练 RMSE | Taylor 验证 RMSE | Wick 验证 RMSE |
|---|---|---:|---:|---:|---:|
| Si | FC2+FC3+FC4 | 0.0400251 | 0.0395405 | 0.0400251 | 0.0395405 |
| SnSe | FC2+FC3+FC4 | 0.00588961 | 0.0162553 | 0.00662811 | 0.0180251 |
| Ba8Ga16Ge30 | FC2+FC3 | 0.0217740 | 0.0264040 | 0.0217740 | 0.0264040 |

RMSE 单位为 eV/Å。

ASR maximum constraint residual：

| 案例 | Taylor | Wick |
|---|---:|---:|
| Si | $4.79\times10^{-16}$ | $7.91\times10^{-16}$ |
| SnSe | $1.54\times10^{-14}$ | $1.50\times10^{-14}$ |
| Ba8Ga16Ge30 | $4.79\times10^{-14}$ | $4.93\times10^{-14}$ |

两种后端均严格满足拟合约束。差异不是 ASR 失败。

以 Wick 结果为分母的 sparse tensor 相对 $L_2$ 差异：

| 案例 | FC2 | FC3 | FC4 |
|---|---:|---:|---:|
| Si | 0.769% | 0.0568% | 36.6% |
| SnSe | 5.52% | 10.6% | 17.9% |
| Ba8Ga16Ge30 | 0.417% | 12.0% | — |

Si 使用 scaled group LASSO。分组正则化不具有基底坐标不变性，因此 Taylor 与 Wick 会选择不同的 FC4，即使它们使用相同数据和物理约束。

SnSe 和 Ba8Ga16Ge30 没有正则化。它们的差异来自当前 Wick 语义：有限 reference covariance、跨阶 contraction、不保存 lowering 产生的 FC1，以及接受 finite-supercell folding residual。这意味着截断后的 Wick predictor 与可转移 Taylor IFC 不保证拥有完全相同的有限数据列空间。Taylor 在这两个案例上取得了更低的拟合误差。

## 物理工作流融合验证

仓库测试进一步锁定：

- FC2–FC4 Wick 参数位于逐阶 ASR null space 时，lowering 后 Taylor 参数仍满足同一 ASR；
- 相同 FC2 经 Born–Huang/Huang 投影后不依赖拟合 provenance；
- SSCHA 使用相同随机样本时，Taylor 与 Wick 的纯 FC2 拟合得到相同有效 FC2；
- SCPH 只消费 Taylor FC2/FC4，修改 `fitted_with` metadata 不改变结果。

对应测试位于：

- `tests/test_fitting_backends.py`
- `tests/test_rotational_sum_rules.py`
- `tests/test_sscha_public.py`
- `tests/test_anharmonic_scph.py`

## 复现

脚本默认依次运行三个真实案例的 Taylor 和 Wick 拟合。完整运行耗时较长，并需要数百 MiB 内存：

```bash
uv run python research/fitting_backends/compare_backends.py
```

只运行一个案例：

```bash
uv run python research/fitting_backends/compare_backends.py --cases si
```

指定结果目录：

```bash
uv run python research/fitting_backends/compare_backends.py \
  --cases snse ba \
  --output /tmp/mlfcs-fitting-bases
```

未指定 `--output` 时，脚本创建独立临时目录。它不会覆盖 `examples/` 中的基线结果。每个 backend 保存 `mlfcs.h5` 和 `metrics.json`，汇总写入 `comparison.json`。

## 结论

ASR、旋转修正、SSCHA 和 SCPH 可以保持 backend-independent。Taylor/Wick 的真实差异来自拟合坐标、正则化和 Wick lowering 的有限超胞语义，而不是约束求解器。当前将 Taylor 作为默认基底、Wick 作为显式实验选择是合理的。
