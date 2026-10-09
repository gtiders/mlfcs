# 长程动力学：待实现的接入设计

本页记录尚未实现的 Γ 方向极限和任意 q 长程组合方案，不是当前可调用接口。已实现的 Ewald 求和与折叠输出见[长程偶极作用](../theory/long-range-electrostatics.md)，使用方法见[长程力与 Ewald](../ewald-api.md)。

## 当前能力与缺口

当前 `DipoleEwald(mapping, born_charges=..., dielectric=...)` 计算三维周期、零宏观电场边界下的谐波长程作用。输出为指定超胞的 folded `CompactForceConstants`，包含 self、onsite 和局域修正；可以扣除训练力并与短程张量相加。

`Harmonic` 当前只接受参数化 `ForceConstants`。`prepare_dynamical_terms` 展开带晶格镜像标签的 FC2，以位置相位规范计算任意分数倒空间 q 点。它没有 Born/介电输入，也不能直接接受相加后的 compact 张量。因此“可以导出长短程总 FC2”还不等于“内部 Harmonic 已经支持极性声子”。

## NAC 的两个范围

### Γ 点方向极限

对零宏观电场的总动力学矩阵，沿非零笛卡尔方向 $q$ 的非解析项为

$$
\Delta D_{i\alpha,j\beta}(\hat q)
=\frac{4\pi C}{\Omega\sqrt{m_i m_j}}
\frac{(q^T Z_i^*)_\alpha(q^T Z_j^*)_\beta}
{q^T\epsilon_\infty q}.
$$

式中的 $q$ 是笛卡尔列向量。这里 $C=\mathrm{Hartree}\times\mathrm{Bohr}$，与当前 Ewald 的 eV、Å 单位一致；质量使用 amu，动力学矩阵单位为 eV/(Å²·amu)。Born 第一轴表示极化方向，第二轴表示位移方向。方向大小相消，但零向量不合法。

分数倒空间行向量应按 $q_{cart}=2\pi q_{frac}A^{-T}$ 转为笛卡尔行向量，再取转置作为式中的 $q$，其中晶胞 $A$ 以行存储；不能直接把分数分量代入介电张量。Γ 点没有唯一方向极限，必须允许用户明确指定方向。此项描述 LO–TO 分裂，不是对 ASR 的替代。

该定义及 Phonopy 的位置相位规范见 [Phonopy formulations](https://phonopy.github.io/phonopy/formulation.html#non-analytical-term-correction)。

### 任意 q 点的长程插值

推荐在现有显式分解下计算

$$
D_{total}(q)=D_{short}(q)+D_{long}(q).
$$

`D_long` 必须包含与扣除训练力时一致的偶极 Ewald、self、局域修正和 onsite；Γ 点按所选电学边界处理，方向极限包含上述非解析项。非零 q 的相位与短程部分保持一致。

有限超胞的 folded 张量丢失独立晶格镜像信息，不能唯一恢复无限晶格任意 q 的 Fourier 和；在超胞相容 q 点上可以通过折叠表达计算，但不能据此宣称支持任意路径插值。

Phonopy 的 Gonze 路线和 Wang 插值是不同方案，不能把 Γ 极限公式直接铺到整个布里渊区当作 Gonze 实现。[官方参考](https://phonopy.github.io/phonopy/reference.html#correction-by-dipole-dipole-interaction)、[API 参数](https://phonopy.github.io/phonopy/phonopy-module.html)。

## 避免重复计入

- 若输入是显式扣除同一 Ewald 后的短程参数模型，补回一次完整 `D_long(q)`。
- 若输入已包含有限超胞长程贡献，则要定义并减去对应的周期长程表达，再补无限晶格长程；不能再直接加一次完整 Ewald。
- 单独的短程 `D_short(Γ)` 加非解析项缺少长程解析部分，不是总模型。
- `CompactForceConstants` 和文件 writer 继续不认识电学模型，不将 Born、介电参数或求和回调塞进通用张量对象。
- 当前外部 NaCl 教程使用 Phonopy NAC；它不等于本库 Harmonic 已实现 NAC。

## 待决定的接口方案

首选明确的短程输入路径，避免推测模型是否已经包含长程作用。以下仅为 API 草案，尚未实现：

```python
from mlfcs.phonon import Harmonic, DipoleEwald

ewald = DipoleEwald(mapping, born_charges=born, dielectric=epsilon)
short_data = data.subtract_forces(ewald.forces(data.displacements))
short_model = FitSystem(short_data).solve()

phonon = Harmonic(short_model, long_range=ewald)
bands = phonon.frequencies(qpoints, gamma_direction=[1, 0, 0])
```

`gamma_direction` 建议明确为笛卡尔方向，只影响 Γ 点极限。Ewald 仍绑定一个 mapping，用于训练力与 compact 输出；未来原胞 q 求和复用其电学参数和同一局域修正。Harmonic 知道长程动力学贡献，ForceConstants 不知道 Ewald。

实现前需要保存或重新组织 Ewald 当前初始化中未保留的局域修正和 onsite 数据；不能只读取 `_tensors`。任意 q 算法还需要证明相位、Γ/self 约定及收敛处理一致。网格约化必须同时保持质量、电学张量和总动力学矩阵对称性；SCPH 接入另作阶段，不随谐波接口隐式启用。

## 实施前的验证要求

验证应覆盖零 Born 退化、各向异性介电、方向缩放与反向不变性、电荷中性下平移零模、非对角晶胞坐标转换、NaCl LO–TO 极限、相容网格与现有 compact 结果一致，以及分裂参数收敛。

与 Phonopy 比较矩阵或频率之前，需要对齐局域修正、相位、单位和电边界，不能只根据同一材料名称要求逐矩阵相等。任意 q 的导数、群速度和输运接口需另外设计。

代码当前不保留初始化中的未折叠局域修正与 onsite 数据，因此不能通过给 Harmonic 增加一个参数就获得正确响应。数据保存方式、总模型的对称群以及未来 SCPH 如何使用长程谐波背景，仍需设计决定。
