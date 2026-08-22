# 仅力数据的力常数拟合

`mlfcs.fitting` 从外部采样的 ASE 结构联合拟合连续的、经过对称性约化的多个阶数。该模块与
有限差分 `sow()` / `reap()` 重建路径相互独立，但两条路径都返回同一种
`ForceConstants` 对象，并共用全部输出 writer。

每个训练结构必须使用参考超胞的原子顺序，并通过 ASE calculator 结果或 `forces` 数组
提供原子力；拟合不使用能量。力模型关于不可约力常数参数保持线性，并采用由位移协方差
正交化的 Wick 特征，缓解相邻阶数之间的效应串扰。

拟合参数保留在 Wick 正交基中。公共 `ForceConstants` 在写出前会精确转换为普通 Taylor
力常数，因为 phonopy、ShengBTE 等格式都把张量解释为 Taylor 导数。设位移协方差为
`Sigma`，换基关系从下式开始：

```text
Phi_T[m] = Phi_W[m] - 1/2 Phi_W[m+2]:Sigma
           + 1/8 Phi_W[m+4]:Sigma:Sigma - ...
```

这只是多项式坐标变换，不是第二次拟合。`FittingResult.parameters` 仍是 Wick 参数；
`FittingResult.force_constants` 则是可供通用格式导出的 Taylor 力常数。

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
)
result = fitter.fit(
    read("train.xyz", index=":"),
    solver="gram",
    batch_size=4,
    validation_split=0.1,
    tolerance=1e-7,
    max_iterations=10_000,
    acoustic_sum_rule=True,
    rotational_invariance=2,
)
result.force_constants.write("FORCE_CONSTANTS_2ND", format="phonopy", order=2)
result.force_constants.write("FORCE_CONSTANTS_3RD", format="shengbte", order=3)
result.force_constants.write("FORCE_CONSTANTS_4TH", format="shengbte", order=4)
```

默认求解器是矩阵无关 LSMR。它以受控 JAX 批次计算 `A @ x` 和 `A.T @ r`，不物化完整
力设计矩阵。逐参数列范数预条件使各拟合阶数的特征处于可比较的数值尺度。`verbose=True`
会打印列尺度估计、算子调用进度以及 LSMR 逐次迭代诊断。

当矩阵无关算子的重复计算成为主要耗时时，可指定 `solver="cached_lsmr"`。MLFCS 会按
JAX 小批次解析构造精确的线性力设计矩阵，将其保存到自动管理的临时磁盘映射，并在
LSMR 全部迭代中复用。缓存路径、存储精度和内部矩阵分块不作为 API 参数暴露，拟合结束
后缓存自动删除。该后端用临时存储和操作系统页缓存换取速度。两种后端的 `batch_size`
均限制为 1--4，它只表示同时处理的结构数。

`solver="gram"` 将每个设计批次流式累积为 `A.T @ A` 和 `A.T @ F`，不保存随快照数增长
的完整设计矩阵。程序按轨道真实的对称图像数和独立参数维数自动精确分组，尺寸桶不是
公开 API 参数。CPU 模式由 JAX 构建设计批次，成熟的 OpenBLAS/SciPy 负责 Gram 累积与
求解；JAX GPU 可用时，设计构建与 Gram 累积都保留在 GPU，完成后只传回一次 Gram。

等式约束通过 `C @ C.T` 的秩揭示伪逆构造隐式零空间投影器，投影共轭梯度始终位于
`null(C)` 内，不再求解不定 KKT 系统后再补偿。`max_iterations` 只是安全上限：求解状态
为零表示投影梯度达到容差，正数表示达到迭代上限但尚未收敛。

主要精度指标采用 ALAMODE 的定义：

```text
相对力误差 = ||F_reference - F_model||₂ / ||F_reference||₂
```

终端以百分比输出，并同时报告单位为 eV/Å 的力 RMSE、验证误差、各阶力贡献 RMS、
LSMR 残差和条件数估计，以及约束 drift。

`rotational_invariance=2` 对应 ALAMODE `ICONST=2`：在笛卡尔坐标中施加已有相邻阶的
旋转约束，但忽略最高阶与下一阶的边界。`rotational_invariance=3` 进一步假定未表示的
下一阶为零并施加最高阶边界；它可能过度约束截断展开，因此绝不作为默认值。刚体旋转
不使用分数坐标，因为非正交晶胞的分数坐标需要额外度量张量，笛卡尔表达才直接对应
物理叉乘与力矩。
