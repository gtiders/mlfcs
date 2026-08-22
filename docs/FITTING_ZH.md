# 仅力数据的力常数拟合

`mlfcs.fitting` 从外部采样的 ASE 结构联合拟合连续的、经过对称性约化的多个阶数。该模块与
有限差分 `sow()` / `reap()` 重建路径相互独立，但两条路径都返回同一种
`ForceConstants` 对象，并共用全部输出 writer。

每个训练结构必须使用参考超胞的原子顺序，并通过 ASE calculator 结果或 `forces` 数组
提供原子力；拟合不使用能量。力模型关于不可约力常数参数保持线性，并采用由位移协方差
正交化的 Wick 特征，缓解相邻阶数之间的效应串扰。

拟合参数保留在 Wick 正交基中。公共 `ForceConstants` 在导出的 FC2--FCn 范围内精确
转换为普通 Taylor 力常数，因为 phonopy、ShengBTE 等格式都把张量解释为 Taylor 导数。设位移协方差为
`Sigma`，换基关系从下式开始：

```text
Phi_T[m] = Phi_W[m] - 1/2 Phi_W[m+2]:Sigma
           + 1/8 Phi_W[m+4]:Sigma:Sigma - ...
```

这只是多项式坐标变换，不是第二次拟合。`FittingResult.parameters` 仍是 Wick 参数；
`FittingResult.force_constants` 则是可供通用格式导出的 Taylor 力常数。

奇数阶 Wick 项还会收缩出 Taylor FC1，即常量力项。FC1 不属于常用声子或 ShengBTE
输出，当前不写入 `ForceConstants`；程序会明确报告其最大分量和净力。因此 FC2--FCn
是该阶数范围内精确的 Taylor 导数，但当遗漏 FC1 非零时，仅靠这些张量不能完整复现
Wick 力模型的常量部分。

非零 FC1 是有意义的诊断信息，本身不代表拟合实现错误。它可能来自：参考结构并非当前
采样分布最适合表示的统计中心或驻点、参考结构残余力、有限快照噪声、非对称采样分布、
多项式最高阶截断，以及空间截断或作用体数截断。平移不变性约束 FC1 的总和，却不要求
每个原子的 FC1 都分别为零。因此 MLFCS 同时报告最大逐原子分量和净值，当前不会静默
把 FC1 强制为零。施加 Taylor FC1=0 会定义一个不同的受约束回归问题，只有经过独立
验证后才适合作为显式选项加入。

平移约束与协方差收缩可交换，因此换基后仍成立。旋转约束会耦合相邻Taylor阶数，不能
直接施加到Wick系数。开启 `rotational_invariance=2` 或 `3` 时，MLFCS先构造Taylor
约束矩阵 `C_T` 和依赖协方差的Wick→Taylor映射 `T(Sigma)`，实际求解约束为：

```text
C_W = C_T @ T(Sigma)
```

最终导出也使用同一个 `T(Sigma)`，因此拟合约束和Taylor输出采用完全一致的定义，而不是
拟合完成后再投影。早期开发实现中曾在未进行此映射时直接开启旋转拟合，这类结果需要
重新计算；仅开启ASR或 `rotational_invariance=0` 的结果不受影响。

```python
from ase.io import read
from mlfcs.fitting import ForceConstantFitter

fitter = ForceConstantFitter(
    primitive=read("POSCAR"),
    reference=read("reference.xyz"),
    supercell=(2, 2, 3),
    orders=(2, 3, 4),
    cutoffs={2: None, 3: 12 * 0.529177210903, 4: 8 * 0.529177210903},
    max_body_orders={2: 2, 3: 3, 4: 3},
)
result = fitter.fit(
    read("train.xyz", index=":"),
    batch_size=4,
    validation_split=0.1,
    tolerance=1e-7,
    max_iterations=10_000,
    acoustic_sum_rule=True,
    rotational_invariance=2,
    allow_unconverged=False,
)
result.force_constants.write("FORCE_CONSTANTS_2ND", format="phonopy", order=2)
result.force_constants.write("FORCE_CONSTANTS_3RD", format="shengbte", order=3)
result.force_constants.write("FORCE_CONSTANTS_4TH", format="shengbte", order=4)
```

`max_body_orders` 可按阶限制一个团簇中不同原子位置的数量。例如 `(0, 0, 1, 1)` 是二体
四阶团簇。某阶省略或设为 `None` 时，保留不超过该力常数阶数的全部体数。
`ForceConstantCalculation` 提供同义的 `max_body_order`，因此拟合与有限差分严格共享同一
相互作用空间定义。

参考结构定义零位移。若参考结构携带力，程序将其视为残余参考力并从每个训练目标中
扣除；未提供参考力时假定为零。除此以外，快照位移和力保持原样：程序不再静默删除
整体平移或快照净力，而是报告最大质心位移、参考力和快照净力，使不一致数据可见但不
替用户修改数据。

拟合器只保留一条求解路径：将每个设计批次流式累积为 `A.T @ A` 和 `A.T @ F`，不保存随快照数增长
的完整设计矩阵。程序按轨道真实的对称图像数和独立参数维数自动精确分组，尺寸桶不是
公开 API 参数。CPU 模式由 JAX 构建设计批次，成熟的 OpenBLAS/SciPy 负责 Gram 累积与
求解；JAX GPU 可用时，设计构建与 Gram 累积都保留在 GPU，完成后只传回一次 Gram。
逐参数精确列范数预条件直接由 Gram 对角元得到。`batch_size` 限制为 1--4，只控制每个
设计批次同时处理的结构数。

等式约束通过 `C @ C.T` 的秩揭示伪逆构造隐式零空间投影器，投影共轭梯度始终位于
`null(C)` 内，不再求解不定 KKT 系统后再补偿。`max_iterations` 只是安全上限：求解状态
为零表示投影梯度达到容差，正数表示达到迭代上限但尚未收敛。默认情况下未收敛会抛出
异常，不生成可写出的力常数；`allow_unconverged=True` 只作为显式诊断开关，并会输出
警告。

主要精度指标采用 ALAMODE 的定义：

```text
相对力误差 = ||F_reference - F_model||₂ / ||F_reference||₂
```

终端以百分比输出，并同时报告单位为 eV/Å 的力 RMSE、验证误差、各阶力贡献 RMS、
投影法方程残差以及约束 drift。

两种旋转模式都在完整的 FC1–FC2 最低恒等式中使用拟合得到的 Taylor FC1，不会静默
把 FC1 强制为零，并在笛卡尔坐标中施加所有已经表示的相邻阶恒等式。`rotational_invariance=2`
采用 ALAMODE `ICONST=2` 的边界取舍，忽略最高阶与下一阶的边界。`rotational_invariance=3` 进一步假定未表示的
下一阶为零并施加最高阶边界；它可能过度约束截断展开，因此绝不作为默认值。刚体旋转
不使用分数坐标，因为非正交晶胞的分数坐标需要额外度量张量，笛卡尔表达才直接对应
物理叉乘与力矩。
