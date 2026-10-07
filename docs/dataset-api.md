# 数据集与三维 Ewald

`ForceDataset` 位于 `mlfcs.dataset`，长程模型 `DipoleEwald` 位于 `mlfcs.phonon.ewald`，也可从 `mlfcs.phonon` 或顶层 `mlfcs` 导入。两者通过位移与力数组衔接，数据集不依赖 Ewald。

拟合和有限差分使用同一个入口：`ForceDataset(mapping, structures)`。数据集接受有序的、已有力结果的 ASE `Atoms`，初始化时收集位移与力；不运行 calculator，不保存采样 metadata，也不保留全部 Atoms。

```python
from ase.io import iread
from mlfcs import ForceDataset, FitSystem

# mapping 已定义原胞模型与实际超胞。
data = ForceDataset(mapping, iread("train.extxyz", index=":"))
system = FitSystem(data, representation="normal")
model = system.solve()
```

## 数据与力扣除

`data.supercell_atoms` 来自 `data.cluster_map.supercell_atoms`，定义位移零点和原子顺序。`displacements` 与 `forces` 都是 `(frames, atoms, 3)` 数组，单位分别为 Å 与 eV/Å。位移取相对于超胞的最近周期像；超出最近像可辨识范围的大位移不应作为谐波训练位移。

原子顺序、晶胞、三维周期性及有限值在收集时检查。不进行原子匹配、帧排序或有限差分预期位移核对。数据集初始化后数组只读；不自动减去参考力、平均力或长程力，也不归一化。

```python
centered = data.subtract_forces(supercell_forces)  # (atoms, 3)，广播给每一帧
corrected = data.subtract_forces(per_frame_forces) # (frames, atoms, 3)
```

只有上述两种形状合法，例如 `(1, atoms, 3)` 不会广播到多帧。扣除返回新数据集，复用位移与 mapping，原始力不修改。两份位移和力数组占用约 `48 * frames * atoms` 字节；Atoms 迭代器只消费一次，但数组会整体驻留内存。normal 拟合仅逐帧处理设计矩阵，不再意味着训练数组流式驻留。

## Born 偶极 Ewald

```python
from mlfcs import DipoleEwald, CompactForceConstants

ewald = DipoleEwald(mapping, born_charges=born, dielectric=epsilon)
long_range = ewald.force_constants()
short_data = data.subtract_forces(ewald.forces(data.displacements))
short_fc = FitSystem(short_data).solve()
short_fc = short_fc.enforce_asr().force_constants
total = CompactForceConstants(short_fc, mapping) + long_range
total.write("force_constants.hdf5", format="phonopy", order=2, storage="hdf5")
```

此模型只适用于三维周期绝缘体系，采用零宏观电场边界；倒易空间的零矢量项被省略。不能作为二维 slab 求和、金属屏蔽模型或固定点电荷的非线性势。

- `born_charges`：原胞顺序，形状 `(N, 3, 3)`，基本电荷单位；两个张量轴依次是极化方向和原子位移方向。诱导偶极为 $p_i=Z_i^*u_i$。
- `dielectric`：Cartesian 电子介电张量，形状 `(3, 3)`，无量纲、对称正定。
- 两者必须与原胞空间群相容，Born 张量满足电中性。维度无关的输入一致性容差为 $10^{-8}$；输入不自动对称化或中性化。
- `rtol=1e-8`、`atol=1e-10`：求和和修正残差容差；`atol` 单位为 eV/Å²。

在屏蔽距离 $\rho(r)=\sqrt{r^T\epsilon^{-1}r}$ 下计算完整实空间、倒易空间与 self 项。内部高斯分裂参数由屏蔽原胞体积决定；扩大两侧截断，直到连续两次 FC2 变化符合容差。该判据是数值收敛检查，不是严格尾项误差证明。

模型为参考超胞附近的谐波能量 Hessian：

$$
E_{\rm LR}=\frac12u^TH_{\rm LR}u,\qquad F_{\rm LR}=-H_{\rm LR}u.
$$

Born 张量和介电张量固定在原胞参考结构上，位移不改变偶极相互作用中心。没有长程 FC3、FC4，也不对质量加权。

### 局域修正与 ASR

一般 Born 张量的中性条件不保证直接补 onsite 后的对称性。依照 [Zhou 等，PRB 100, 184309，§II](https://arxiv.org/pdf/1805.08903) 的局域修正框架，使用当前 `ClusterSpace` 的全部非 onsite FC2 orbit 为支持空间。cutoff 不自动扩大。

先计算未修正偶极行和的反对称部分 $Q$，求解 $M\theta_{\rm cor}=-Q$。使用展开后各 orbit image 张量的 Frobenius 范数作为目标；实现通过块度量白化与 SVD 获得最小范数解，与论文使用的稀疏解选择不同。修正自身遵循空间群和张量交换对称性。

随后定义真正的 onsite 项：

$$
\Phi^{\rm LR}_{i,i}(0)=-\sum_{(j,R)\ne(i,0)}\Phi^{\rm LR}_{i,j}(R).
$$

非零平移的同一个 primitive site 属于相互作用，不是 onsite；先在晶格层定义修正与 onsite，再折叠到超胞。修正只改变现有 cutoff 内的作用与 onsite，不改变远距离偶极尾。

`ewald.correction_norm` 表示展开修正张量的 Frobenius 范数；`ewald.asr_residual` 是折叠后最大行和绝对值。没有 FC2 或支持空间不足会报错；不回退到未修正版本。初始化检查折叠后的 ASR 与交换对称性。长程部分自身满足 ASR，因此短程仍可使用现有齐次 ASR。这里不额外施加 Born–Huang 旋转约束。

扣除与回加必须使用同一个模型；改变修正支持空间会改变长短程分解，需要重新拟合短程部分。

onsite 由完整 Ewald 行和及局域修正决定，并不是将长程张量投影到现有 onsite basis。依赖 cutoff 的是局域修正支持空间，所以 onsite 也会随这份分解改变。远距离偶极尾保持不变；满足 ASR 并不等于有限 cutoff 的总模型已经收敛，仍应比较不同 cutoff 下重新拟合并回加后的总力、声子频率等物理量。

## 绑定 mapping 与统一张量

Ewald 初始化直接接受 mapping，以其超胞为参考几何，计算原胞层修正、onsite 和该超胞的长程 compact FC2。随后力计算与张量输出复用结果：

```python
forces = ewald.forces(data.displacements)
long_range = ewald.force_constants()
```

一个 `DipoleEwald` 对象只对应一个明确的 mapping。重复调用不会重新求和；换超胞时构造另一个对象。位移数组必须采用该 mapping 的原子顺序，调用者负责配对。没有可变的当前超胞或全局缓存。

`ewald.forces(...)` 与 `long_range.harmonic_forces(...)` 使用同一套 FC2 收缩，均不计算高阶非线性力。数组读取由通用张量对象的 compact_array/full_array 提供；Ewald 不保留另一套数组展开接口。

短程参数模型和已折叠数组使用同一个类型：

```python
short = CompactForceConstants(short_fc, mapping)
explicit = CompactForceConstants({2: compact_fc2}, mapping)
total = short + long_range
```

显式数组只接受声明阶数对应的 primitive-first 布局，不根据 shape 猜测 full Hessian。相加要求同一目标超胞与原子顺序；用户负责物理配对，形状和有限值在输入处检查。Ewald 输出不持有 Ewald 对象，张量与 IO 层也不执行 Ewald 求和。

## FC2 数组与分块读取

```python
compact = long_range.compact_array(order=2)  # (N, n, 3, 3)，显式物化
full = long_range.full_array(order=2)        # (n, n, 3, 3)，显式物化
forces = long_range.harmonic_forces(data.displacements)

for atom_slices, tensors in total.compact_blocks(order=2):
    consume(atom_slices, tensors)
```

输出是能量二阶导数，单位 eV/Å²，符号不是力 Jacobian。compact 首轴对应原胞各 site 的零平移代表；full 两个原子轴遵循 `mapping.supercell_atoms` 顺序。力计算不物化 full。

对于 atom-major Cartesian Hessian，转换为：

```python
full = H.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
H = full.transpose(0, 2, 1, 3).reshape(3*n, 3*n)
```

`compact_blocks` 和 `full_blocks` 按原子轴分块，每个 Cartesian 张量保持完整。默认 `max_bytes=8*1024**2` 为数值工作区预算，涵盖当前输出、一次消费者复制、输出标签及张量动作暂存，不包含已有来源数组与标签索引。生成的块不会被下一次迭代覆盖；调用者若收集所有块，则自行承担全量内存。显式 array 方法另外检查完整结果的可表示范围，不承诺物理内存一定足够。

原生 HDF5 仍只保存参数化模型；保存剩余力拟合结果时，保存的是短程部分，不会自动保存 Ewald。CompactForceConstants 没有 save/load。

## 外部导出与晶格信息

`total.write(...)` 与 `ForceConstants.write(...)` 使用同一个格式 writer。Phonopy FC2 文本/HDF5 和 Phono3py FC3 HDF5 逐块输出 full 文件布局，不先物化 full 或完整 compact 数组。清理阈值在所有来源和周期 aliases 相加后应用，单位为对应阶数的力常数单位。

短程参数来源保留未折叠的 sites 与相对晶格平移，支持 `lattice_blocks(order)` 和 ShengBTE FC3/FC4。已折叠数组只知道周期镜像贡献之和，不能恢复每个独立平移的张量。

```python
total.write("FORCE_CONSTANTS_3RD", format="shengbte", order=3)
```

这在短程模型包含 FC3 时成立，因为 Ewald 只贡献 FC2；FC4 同理。总 FC2 可以直接写出 Phonopy，无需恢复独立镜像标签。若直接请求没有来源信息的总 FC2 的 lattice_blocks，则报错，不会自动分配到最近镜像。外部格式只支持 Phonopy、Phono3py 和 ShengBTE。

折叠超胞数组不能恢复任意波矢的长程作用，也不应再叠加一份已经计入的长程 FC2。完整 LO–TO 插值仍需要单独的倒易空间长程模型；目前不提供该接口。


## 每个构型使用不同的电学数据？

当前 `DipoleEwald` 使用一套参考原胞 Born/介电张量。Born 电荷是极化对位移的导数，不是每个原子的静态点电荷；不能把 Bader/Mulliken 电荷直接代入这一接口。

逐帧更换谐波核然后写 $F_t=-H_tu_t$，仅定义了一套逐帧扣除规则。若 $H$ 随位移变化，则对 $E(u)=u^TH(u)u/2$ 求导还包含 $\partial H/\partial u$ 项；单独给出每帧 Born/介电张量不足以确定一般构型依赖模型的完整力。当前接口不声称实现这一功能。

外部若能提供一致定义的长程力数组，仍可通过 `data.subtract_forces(per_frame_forces)` 扣除，但此时不能自动把扣除结果回加为一份固定的长程 FC2。
