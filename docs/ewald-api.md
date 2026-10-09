# 长程力处理：扣除 Ewald 偶极贡献并回加 FC2

极性绝缘体中的位移会产生偶极响应，它引起的力延伸到很远的晶格位置。如果直接用有限 cutoff 的模型解释全部原子力，短程参数可能不得不吸收这部分长程贡献，结果也会随截断和超胞变化。

一种处理方式是先计算已知的长程谐波力，从目标力中扣除，再恢复剩余的短程力常数。需要总 FC2 时，将同一份长程贡献加回来。MLFCS 的 `DipoleEwald` 为这条流程提供 Born 有效电荷与电子介电张量定义的三维偶极模型。

本章从[核心概念](core-concepts.md)中的 mapping 和数据集出发，依次介绍电学输入、力扣除、短程模型恢复及总张量导出。Ewald 求和的推导见[长程电静力学](theory/long-range-electrostatics.md)。

## 1. 先确认这份长程模型描述什么

`DipoleEwald` 计算的是参考晶体附近的谐波偶极作用，不是一般点电荷势。Born 电荷和介电响应固定在参考几何上，位移不会改变相互作用中心或重新生成电学参数。

对于超胞位移 $u$，它给出

$$
F_{\rm LR}=-\Phi_{\rm LR}u.
$$

这里 $\Phi_{\rm LR}$ 是能量的二阶导数，单位 eV/Å²；位移用 Å，力用 eV/Å，不做质量加权。当前没有长程 FC3、FC4 或其他非线性力项。

求和适用于三维全周期绝缘体系。倒易零矢量被省略，对应零宏观电场边界；它不是二维 slab 求和，也不是金属屏蔽模型。该边界条件也不等于已经加入声子 Γ 点方向极限的 NAC。

## 2. 用 mapping 和电学数据构造模型

```python
from mlfcs import DipoleEwald

ewald = DipoleEwald(
    mapping,
    born_charges=born,
    dielectric=epsilon,
    rtol=1e-8,
    atol=1e-10,
)
```

构造的第一个参数 `cluster_map` 必须是 `ClusterMap`。它同时指定模型的参考胞、位点顺序、局域 FC2 支持空间和要实现的超胞。空间必须包含二阶，即使还没有拟合出 FC2 系数也可以先建立 Ewald 模型。

初始化完成时，所选超胞的长程 FC2 已经算好。换超胞或电学数据应重新构造对象，不能只替换输入位移数组来改变 mapping。

### Born 电荷的顺序、单位与轴

`born_charges` 是必需关键字，形状为 `(N, 3, 3)`，`N = mapping.cluster_space.n_atoms`。第一个轴遵循模型参考胞的原子顺序，而不是超胞顺序；非最小参考胞也应为其每个位点提供一份张量。

`born_charges[i, alpha, beta]` 的两个 Cartesian 轴分别是极化方向和位移方向，单位为基本电荷。诱导偶极满足 $p_i=Z_i^*u_i$。从其他程序读入时，要核对轴定义、晶体取向和原子编号，不能只根据形状判断配套性。

Born 电荷是极化对位移的导数，不是静态原子电荷，不能直接用 Bader 或 Mulliken 电荷替代。输入必须有限、满足电中性，并与参考结构的空间群操作相容。

### 电子介电张量

`dielectric` 也是必需关键字，是同一 Cartesian 坐标系中的 `(3, 3)` 电子介电张量，无量纲。它必须有限、对称、正定，并与空间群相容。这里使用电子介电响应，不应把包含离子响应的静态总介电常数直接代入。

当前代码要求张量转置前后元素完全相同；电中性和空间群一致性使用固定的 $10^{-8}$ 输入容差及相应尺度检查。输入不会自动中性化、对称化或旋转到另一晶胞取向。

### 求和容差

`rtol` 默认 `1e-8`，必须有限且位于 `(0, 1)`；`atol` 默认 `1e-10`，必须正且有限，单位为 eV/Å²。两者控制求和收敛和约束残差检查，不是拟合的停止参数。

内部在介电屏蔽度量下累加实空间、倒易空间与 self 项。高斯分裂尺度由屏蔽参考胞体积决定，同时扩展两侧求和范围，直到连续两次 FC2 变化满足容差。没有用户需要手动选择的分裂参数或求和半径；这个判据是数值收敛检查，不是严格的尾项误差证明。

mapping 类型错误会抛出 `TypeError`。没有 FC2、输入形状或物理条件不合法、容差不合法会抛出 `ValueError`；求和扩展后仍未收敛会抛出 `RuntimeError`。非有限求和结果或不支持的数值范围也会显式失败。

## 3. 局域修正为什么会用到 cutoff

Ewald 偶极尾没有被 cluster cutoff 截断。cutoff 限制的是一份额外的**局域修正**：它使用现有非 onsite FC2 轨道，协调长程项的行和与 onsite 对称性，再由声学求和规则确定 onsite 项。

局域修正最小化展开 Cartesian 张量的 Frobenius 范数，不是最小化参数向量范数，也不是稀疏拟合惩罚。该构造沿用 [Zhou 等的长短程分解框架](https://doi.org/10.1103/PhysRevB.100.184309)，具体求解与理论对应见[长程电静力学](theory/long-range-electrostatics.md)。

因此，改变 FC2 cutoff 会改变局域支持与长短程分解，onsite 也会随之变化；远距离偶极尾保持。每次改变这份分解，都应重新扣除、拟合和回加，比较总结果，而不是只比较短程系数。

局域支持不足以恢复所需条件时抛出 `ConstraintProjectionError`，不会自动扩大 cutoff 或返回未经修正的版本。初始化还检查最终超胞 FC2 的 ASR 和交换对称性。这里不施加 Born–Huang 旋转约束。

Ewald 自身的局域与 onsite 修正不依赖 `ClusterSpace.asr` 开关，长程 FC2 会单独满足 ASR。若总力来自平移不变的模型，短程恢复通常也使用 `ClusterSpace(..., asr=True)`；该开关约束短程拟合或差分，不是启动 Ewald 求和的开关。

### 构造后查看什么

| 公开属性 | 含义 |
|---|---|
| `cluster_map` | 对象绑定的具体超胞 mapping。 |
| `born_charges`、`dielectric` | 输入顺序与坐标系中的电学张量，保存为独立只读数组。 |
| `rtol`、`atol` | 使用的求和与残差容差。 |
| `correction_norm` | 展开后的非 onsite 局域修正 Frobenius 范数，不包含单独确定的 onsite ASR 项，单位 eV/Å²。 |
| `asr_residual` | 修正并折叠后的 FC2 最大行和绝对残差，单位 eV/Å²。 |

较小 ASR 残差说明约束检查通过，不证明物理模型、截断或训练集已经收敛。

## 4. 对数据集计算长程力并扣除

```python
long_forces = ewald.forces(dataset.displacements)
short_data = dataset.subtract_forces(long_forces)
```

`forces(displacements)` 接受一帧 `(n, 3)` 或一批 `(frames, n, 3)` Cartesian 位移，`n = mapping.n_atoms`。返回同形状力数组；非法形状或非有限输入被拒绝。位移原点与原子顺序须对应绑定的 mapping，调用者负责配对。

这里传的是位移，不是绝对位置或 ASE 结构。通过 `ForceDataset` 获得位移可以保持统一的最小像约定；它适合参考晶体附近的构型，不表示扩散轨迹的解包位移。

这一步复用预先计算的 FC2，不再次执行 Ewald 求和。`subtract_forces()` 返回新数据集，保持 mapping、位移和帧顺序，原始数据不变。多帧力按帧扣除；单个 `(n, 3)` 数组可广播扣除，但 `(1, n, 3)` 不会广播到多帧。

数据集读取已保存的 ASE 力，不启动 calculator，也不自动扣除任何贡献。迭代器可以避免先保存全部 `Atoms`，但位移与力数组仍整体驻留内存，两份数组约占 `48 * frames * n` 字节。详见[核心概念](core-concepts.md)。

### 拟合与有限差分怎样衔接

```python
from mlfcs import FitSystem

short_fc = FitSystem(short_data, representation="raw").solve()
```

拟合此时使用剩余力，返回的 `short_fc` 是短程参数化模型。如果数据来自有限差分计划，可以改为：

```python
# dataset 必须按这个 fd.sow() 的采样顺序收集。
short_fc2 = fd.reap(short_data)
```

扣除不改变差分步长与帧顺序，因此零步长外推仍按原计划进行。重建出的却是剩余力对应的参数；使用时要明确它不是已经回加长程项的总模型。

## 5. 回加长程 FC2，再导出总结果

```python
from mlfcs import CompactForceConstants

long_range = ewald.force_constants()
short_range = CompactForceConstants(short_fc, mapping)
total = short_range + long_range

total.write("force_constants.hdf5", format="phonopy", order=2, storage="hdf5")
```

`force_constants()` 无参数，返回当前超胞的 `CompactForceConstants`，包含周期像、self、局域修正与 onsite 贡献。它不重复求和，返回张量对象也不持有或调用 Ewald 引擎。

相加应使用与力扣除时相同的长程对象、超胞和原子顺序。用户负责物理布局配套；相同形状不能证明两个来源属于同一个问题。`long_range.harmonic_forces(dataset.displacements)` 与 `ewald.forces(...)` 使用同一套 FC2 收缩，可以用于核对分解后的力。

保存 `short_fc.save(...)` 得到的是原生短程参数模型，不会自动保存 Ewald 或总模型。`CompactForceConstants` 没有原生 `save()`／`load()`，总张量应使用外部格式导出；重建长程处理时还需保留参考结构、mapping、电学输入和容差等信息。

### 需要数组时，用 compact 还是 full

```python
compact = long_range.compact_array(order=2)  # (N, n, 3, 3)
full = long_range.full_array(order=2)        # (n, n, 3, 3)
```

两种数组都显式物化，单位为 eV/Å²。compact 首轴是参考胞各位点的零平移代表，第二轴按超胞原子顺序；full 两个原子轴都按超胞顺序。它们是能量 Hessian 分块，不是力对位移 Jacobian，后者多一个负号。

若外部程序需要 `(3*n, 3*n)`、原子优先的矩阵布局，转换为 `H = full.transpose(0, 2, 1, 3).reshape(3*n, 3*n)`。反向转换为 `H.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)`。显式数组输入可用 `CompactForceConstants({2: compact}, mapping)`，但它只接受声明阶数的 compact 布局，不根据形状猜测 full。

大超胞可以分块读取：

```python
for atom_slices, tensors in total.compact_blocks(order=2):
    # 每个 tensors 是当前原子切片上的完整 Cartesian 张量块。
    print(atom_slices, tensors.shape)
```

`full_blocks` 提供 full 布局的同类读取。二者的 `max_bytes` 默认 `8*1024**2`，控制工作区预算，含输出、一次消费者复制和动作暂存，不包含已保存的来源数组与标签索引。收集全部块仍需全量内存。力计算不物化完整 full 数组，但 Ewald 初始化已保存该超胞的 compact FC2，不能据此称其完全不物化长程张量。

### 外部格式与晶格镜像信息

`total.write(file, *, format, order, storage=None, threshold=1e-8)` 使用通用格式 writer。`order` 显式选择已有阶数；`format` 可选 `"phonopy"`、`"phono3py"` 或 `"shengbte"`。Phonopy 支持 FC2，默认文本也可用 HDF5；Phono3py 支持 FC3 HDF5；ShengBTE 支持 FC3、FC4 文本。不合法组合会报错，写出成功返回目标路径。

`threshold` 是对应阶数物理单位中的非负清理阈值，在来源和周期混叠贡献相加后应用。Phonopy／Phono3py 周期张量逐块输出，无需先构造完整 full 数组。

短程参数来源保留独立的晶格平移，Ewald 输出则已经折叠到有限超胞。后者不能恢复每个镜像的独立贡献，因此请求含这类 folded 来源的 FC2 `lattice_blocks()` 会报错，不会擅自把贡献分给最近镜像。

若短程模型含 FC3 或 FC4，总对象仍可按这些阶数导出 ShengBTE，因为 Ewald 不贡献这些阶数：

```python
# 要求 short_fc 确实包含 FC3。
total.write("FORCE_CONSTANTS_3RD", format="shengbte", order=3)
```

## 6. 总 FC2 不等于任意 q 的长程模型

已折叠 FC2 能计算指定超胞内的谐波力并导出周期张量，但单凭它不能恢复任意 q 的解析长程响应。当前 `Harmonic` 也不接收 Ewald 或总 compact 张量作为额外谐波背景。

极性材料的声子插值和 LO–TO 分裂还需要相容的 NAC 处理。已有长程 FC2 不应再未经分解检查就重复叠加；当前外部 Phonopy 流程见 [NaCl 教程](notebooks/nacl-long-range.ipynb)。

每帧使用不同 Born／介电张量也不属于当前接口。若谐波核随位移变化，仅写 $F=-H(u)u$ 会遗漏能量求导中的核变化项。外部若能给出一致定义的逐帧长程力，仍可用数据集扣除，但不能因此自动回加成一份固定 FC2。

## 7. 从已有电学输入到总 FC2 的完整例子

下面假设已有参考结构、带力样本，以及与参考胞顺序和 Cartesian 取向一致的 `born.npy` 和 `epsilon.npy`。这些文件中的数据须来自相应材料计算；示例不预设通用的 Born 电荷或介电常数。

```python
import numpy as np
from ase.io import read, iread
from mlfcs import (
    ClusterSpace, ClusterMap, ForceDataset, FitSystem,
    DipoleEwald, CompactForceConstants,
)

reference = read("POSCAR")
space = ClusterSpace(
    reference,
    cutoffs={2: 4.0},
    max_body_orders={2: 2},
    asr=True,
)
mapping = ClusterMap(space, read("SPOSCAR"))
mapping.rank_info().require_full()
dataset = ForceDataset(mapping, iread("train.extxyz", index=":"))

ewald = DipoleEwald(
    mapping,
    born_charges=np.load("born.npy"),
    dielectric=np.load("epsilon.npy"),
    rtol=1e-8,
    atol=1e-10,
)
print("局域修正范数：", ewald.correction_norm, "eV/Å²")
print("长程 ASR 残差：", ewald.asr_residual, "eV/Å²")

short_data = dataset.subtract_forces(ewald.forces(dataset.displacements))
system = FitSystem(short_data, representation="raw")
short_fc = system.solve()
print("剩余力训练 RMSE：", system.rmse(short_fc), "eV/Å")
short_fc.save("short-range.mlfcs")

total = CompactForceConstants(short_fc, mapping) + ewald.force_constants()
total.write("force_constants.hdf5", format="phonopy", order=2, storage="hdf5")
```

cutoff、超胞和求和容差仍需收敛检查。具体材料、电学输入和 NAC 后续计算的执行结果见 [NaCl 教程](notebooks/nacl-long-range.ipynb)。

整条流程应使用同一份长短程分解：先按参考电学数据计算长程力，从观测中扣除，恢复短程模型，最后按同一 mapping 回加长程 FC2。拟合保存的是短程参数，外部导出得到的是指定超胞上的总张量，两者都应清楚标记。
