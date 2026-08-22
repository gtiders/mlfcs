---
title: 仅力数据的力常数拟合
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

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

程序会检查空间截断和作用体数支撑域在所有同奇偶阶 Wick 收缩下是否闭合。只有当缺失目标
簇的位点对称允许张量空间为零，并且所有对称图像聚合后的收缩在尺度化数值容差内为零时，
该缺失簇才可合法跳过。若对称禁止簇得到非零收缩，程序会报告协方差对称化、周期图像或聚合
错误；若缺失簇存在非零对称允许空间，则报告真正的支撑域闭合错误。两类贡献都不会被静默丢弃。

非零 FC1 是有意义的诊断信息，本身不代表拟合实现错误。它可能来自：参考结构并非当前
采样分布最适合表示的统计中心或驻点、参考结构残余力、有限快照噪声、非对称采样分布、
多项式最高阶截断，以及空间截断或作用体数截断。平移不变性约束 FC1 的总和，却不要求
每个原子的 FC1 都分别为零。因此 MLFCS 同时报告最大逐原子分量和净值，当前不会静默
把 FC1 强制为零。施加 Taylor FC1=0 会定义一个不同的受约束回归问题，只有经过独立
验证后才适合作为显式选项加入。

驻点判断必须使用 Wick→Taylor 换基后报告的 Taylor FC1，不能直接使用求解器内部的
Wick 一阶系数；高阶奇数 Wick 项会收缩到 Taylor FC1，因此两者一般不同。在固定晶格下，
可将同一 primitive site 的所有超胞像聚合，并在去除三个整体平移零模后，用 FC2 的
伪逆估算局部 Newton 修正 `Delta u = -Phi2^+ Phi1`。该位移只适合作为参考结构驻点和
训练数据质量的诊断，只有远小于训练位移范围时才可视为可信的局部修正建议。它不能让
各超胞像独立弛豫，不能判断晶格常数或晶胞形状是否最优，也不能替代带应力的第一性原理
结构优化；移动参考结构后通常需要重新计算力并重新拟合。

平移约束与协方差收缩可交换，因此仍在拟合中施加。Born-Huang 与 Huang 不作为 Wick
拟合期约束：在联合 FC2/FC4 Wick 拟合中约束最终 Taylor FC2 会连带改变 FC4。拟合完成后
对 `result.force_constants` 调用独立的 FC2 后处理；它不会改动任何高阶。

```python
from ase.io import read
from mlfcs.fitting import ForceConstantFitter

fitter = ForceConstantFitter(
    primitive=read("POSCAR"),
    reference=read("reference.xyz"),
    orders=(2, 3, 4),
    cutoffs={2: 8.0, 3: 12 * 0.529177210903, 4: 8 * 0.529177210903},
    max_body_orders={2: 2, 3: 3, 4: 3},
)
result = fitter.fit(
    read("train.xyz", index=":"),
    batch_size=4,
    validation_split=0.1,
    tolerance=1e-7,
    max_iterations=10_000,
    acoustic_sum_rule=True,
    allow_unconverged=False,
)
write_force_constants(result.force_constants, "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
write_force_constants(result.force_constants, "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
write_force_constants(result.force_constants, "FORCE_CONSTANTS_4TH", format="shengbte", order=4)
```

MLFCS 不支持将外部低阶 IFC 冻结到高阶拟合中。所有阶数必须在同一个 Wick 参数空间中联合
确定，以保证高阶收缩到低阶的关系、对称性和约束保持一致。

`max_body_orders` 可按阶限制一个团簇中不同原子位置的数量。例如 `(0, 0, 1, 1)` 是二体
四阶团簇。某阶省略或设为 `None` 时，保留不超过该力常数阶数的全部体数。
`ForceConstantCalculation` 提供同义的 `max_body_order`，因此拟合与有限差分严格共享同一
相互作用空间定义。

参考结构定义零位移。若参考结构携带力，程序将其视为残余参考力并从每个训练目标中
扣除；未提供参考力时假定为零。除此以外，快照位移和力保持原样：程序不再静默删除
整体平移或快照净力，而是报告最大质心位移、参考力和快照净力，使不一致数据可见但不
替用户修改数据。

拟合器将每个设计批次流式累积为 `A.T @ A` 和 `A.T @ F`，不保存随快照数增长
的完整设计矩阵。程序按轨道真实的对称图像数和独立参数维数自动精确分组，尺寸桶不是
公开 API 参数。周期几何只保存原子索引，笛卡尔分量组合在 JAX 核内即时生成，不再对
每个轨道、对称图像、平移和张量分量预展开。每个核只返回自身的局部参数列；协方差和
轨道大数组作为运行时参数传入，不再被捕获为 XLA 常量。

无正则拟合会在设计累计前参数化硬约束。程序按约束连通分块执行带主元 QR，构造稀疏
映射 `Z` 并令 `theta = Z q`，随后直接累计 `(A Z).T @ (A Z)`。因此 Gram 存储和
求解规模取决于约束后的自由度，而不是原始不可约参数数目；程序不会构造全局稠密零空间。
每次拟合会创建一个 `PreparedDesignProgram`：它只打包一次轨道 tile、上传一次静态缓冲区、
缓存 JIT callable，训练、验证和诊断均复用同一对象。CPU 模式由 JAX 构建物理设计 tile，
OpenBLAS/SciPy 完成稀疏约化和 Gram 累积。JAX GPU 模式中，物理设计、有界稀疏零空间
约化和 Gram 累积均驻留在设备端，只在最后取回充分统计量。
逐参数精确列范数预条件直接由 Gram 对角元得到。`batch_size` 限制为 1--4，只控制每个
设计批次同时处理的结构数。

`fit(..., cache_directory="路径")` 是稳定公开的 Gram 恢复缓存 API。MLFCS 对位移、力、
协方差和参数化输入生成指纹，并在 `路径/gram-<fingerprint>/` 保存完成的 Gram 统计量；
相同输入会复用统计量，任一输入改变则自动使用不同缓存条目。结果的
`FittingResult.cache_directory` 返回实际使用的条目；未启用缓存时为 `None`。

开发时可设置 `MLFCS_JAX_TRANSFER_GUARD=log` 或 `disallow` 审计意外的隐式 JAX 数据
传输；默认不启用。旋转约束使各阶混合时，逐阶力 RMS 使用一次共享的拟合后特征遍历，
不会按阶重复生成 Wick 特征。

默认的 `regularization=None` 求解严格受约束的无正则 Gram 问题；设置
`regularization="scaled_group_lasso"` 后复用同一 Gram 统计量，以 ADMM 求解联合估计残差
尺度的组稀疏问题。每一组对应一个完整的空间群不可约团簇轨道，因此程序只会整体保留或抑制
一个物理相互作用轨道，不会任意挑选轨道内部的张量分量。阈值会修正轨道维数，残差噪声尺度和
正则强度在优化中自动估计，不要求用户提供惩罚参数或执行交叉验证。两种模式中的 ASR 和可选
旋转恒等式始终是硬等式约束。

默认无正则求解使用上述显式分块稀疏参数化。轨道组 LASSO 仍使用基于 `C @ C.T`
秩揭示伪逆的隐式零空间投影器，因为一般零空间变换会破坏其逐轨道局部惩罚分组。两条
路径的迭代都始终位于 `null(C)` 内，不再求解不定 KKT 系统后再补偿。`max_iterations` 只是安全上限：求解状态
为零表示投影梯度达到容差，正数表示达到迭代上限但尚未收敛。默认情况下未收敛会抛出
异常，不生成可写出的力常数；`allow_unconverged=True` 只作为显式诊断开关，并会输出
警告。

主要精度指标直接由参考力和模型力定义：

```text
相对力误差 = ||F_reference - F_model||₂ / ||F_reference||₂
```

终端以百分比输出，并同时报告单位为 eV/Å 的力 RMSE、验证误差、各阶力贡献 RMS、
投影法方程残差以及约束 drift。

物理 FC2 的 Born-Huang/Huang 修正使用
`enforce_rotational_sum_rules(result.force_constants, ...)`。默认 `strength=1.0` 为严格
模式；详见[求和规则]。
