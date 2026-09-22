---
title: 有限差分 API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 有限差分 API

## `FiniteDifferenceCalculation`

```python
FiniteDifferenceCalculation(
    atoms: Atoms,
    *,
    order: int,
    reference: Atoms,
    cutoff: float | int = -5,
    max_body_order: int | None = None,
    displacement: float = 0.01,
    symprec: float = 1e-5,
)
```

| 参数 | 含义 |
|---|---|
| `atoms` | primitive ASE `Atoms`。首个位置参数不是 reference。 |
| `order` | 目标 IFC 阶数，必须至少为 2。一次对象只重建一个阶。 |
| `reference` | 显式训练/位移超胞，决定原子顺序和可辨识性。 |
| `cutoff` | 必填：正数表示 Å 半径，负整数表示 primitive 邻居壳层。不接受 `None`；参考超胞只决定模型是否可辨识，不会缩短模型。 |
| `max_body_order` | cluster 中允许的最大不同 `(site,R)` 数；`None` 表示不额外限制。 |
| `displacement` | 中心差分基础步长，单位 Å，默认 0.01。 |
| `symprec` | 结构对应与空间群容差。 |

构造阶段建立 interaction/orbit space；首次访问 `plan`、`manifest` 或调用 `sow()` 时才生成对称约化位移计划。

## 计划身份与指纹

位移构型与力之间可能相隔数小时甚至数天（外部 DFT、NEP 势、批处理队列），期间持有计划的 Python
对象通常已经不存在。仅凭"构型数量 + 从零开始的编号"无法判断力属于哪个计划：同一晶体、同一 order、
同一 cutoff 和同一位移幅度的两个 plan 可以要求**相同数量**的构型，却观测**不同的分量行**。

因此每个计划都用 `DisplacementManifest` 描述自己：

| 字段 | 内容 |
|---|---|
| `schema_version` | manifest 文档与布局版本，参与哈希。 |
| `primitive_fingerprint` / `reference_fingerprint` | primitive 与 reference 的结构指纹。 |
| `order`、`cutoff_angstrom`、`max_body_order`、`symprec`、`displacement_angstrom` | 模型与差分设置。 |
| `derivative_backend`、`stencil_signs`、`extrapolation` | 微分后端、stencil 符号与零步长外推描述。 |
| `orbits` | 每个规范 orbit 的 `representative`、`dimension`、`observation_rows`、整数 `exact_lattice_basis` 与 `images`。 |
| `displacement_keys` | 排序后的观测键。 |
| `configurations` | 每个构型的 id、key、位移原子、方向、符号与步长。 |

`fingerprint` 是对上述字段（除指纹自身）按 `sort_keys=True`、`separators=(",", ":")` 序列化后的
规范 JSON 文档做 SHA-256；浮点一律用 `float.hex()` 编码，因此平台、字节序与浮点转十进制算法都不影响
结果，`repr()`、对象哈希、内存布局与浮点原始字节都不会进入指纹。结构指纹同样按规范文档计算，包
含原子序数、cell、wrap 后的分数坐标与周期性；身份是精确的：坐标相差最后一位就属于不同计划。

`DisplacementManifest.save()`/`load()` 可往返序列化；读取时会重新计算指纹，被手工编辑过的文件会被
拒绝。

## `sow()`

```python
sow() -> DisplacementBatch
```

返回 `DisplacementBatch`：可迭代、可 `len()`，其 `structures` 是位移后的 ASE `Atoms`，每帧的
`info` 中带 `mlfcs_configuration_id`（从零开始）与 `mlfcs_plan_fingerprint`。跨机器/跨天工作流应先
保存 manifest：

```python
batch = calculation.sow()
batch.manifest.save("displacements.json")
for atoms in batch:
    write(f"disp-{atoms.info['mlfcs_configuration_id']:04d}.vasp", atoms, vasp5=True)
```

## `reap()`

```python
reap(
    forces: ForceBatch,
    *,
    acoustic_sum_rule: bool = True,
    asr_tolerance: float = 1e-10,
) -> ForceConstants
```

`reap()` **只**接受 `ForceBatch`，其字段为 `fingerprint`、`configuration_ids`、`forces` 与
`schema_version`。`configuration_ids` 可以按任意顺序给出，接收方会先按 id 还原顺序再求导；
`forces` 形状必须为 `(n_configurations, n_reference_atoms, 3)` 且全部有限。

以下情况一律在求导前失败，并给出指名具体差异的异常：

- 传入裸 `ndarray`、按位置排列的 `list`/`tuple`，或以数字 configuration id 为键的 `Mapping`：
  它们都不携带计划指纹，异常会提示用 `sow()` 重新生成位移；
- 指纹或 schema 版本与计划不符；
- configuration id 缺失、重复或未知；
- 原子数或力 shape 与计划不符。

不存在"缺少指纹就按位置继续"的回退：旧 `sow()` 产生的裸序列无法被新计划静默接收，必须重新生成全部
位移构型。

## 外部 calculator 工作流

VASP、Quantum ESPRESSO 等外部程序返回的是落在磁盘上的力，因此先用 manifest 记住计划，再用
`ForceBatch` 把力交回来：

```python
from mlfcs.finite_difference.plan_identity import DisplacementManifest, ForceBatch

manifest = DisplacementManifest.load("displacements.json")   # 播种时保存
ids, forces = [], []
for index in range(len(manifest.configurations)):
    ids.append(index)
    forces.append(read_forces_from_external_code(index))      # (n_reference_atoms, 3)
result = calculation.reap(
    ForceBatch(
        fingerprint=manifest.fingerprint,
        configuration_ids=tuple(ids),
        forces=np.asarray(forces),
    )
)
```

只要 `calculation` 仍由同一 primitive、reference、order、cutoff、body order、symprec、位移与微分
后端构造，指纹就会重建为同一值；任何一项改变都会改变指纹，从而拒绝旧力。直接把 ASE
`Calculator` 交给 `run()` 的内部工作流自动完成同样的绑定，不需要手写 `ForceBatch`。

## `evaluate()` 与 `run()`

```python
evaluate(
    calculator: Calculator,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> ForceBatch

run(
    calculator: Calculator,
    *,
    progress: Callable[[int, int], None] | None = None,
    acoustic_sum_rule: bool = True,
    asr_tolerance: float = 1e-10,
    derivative_backend: Literal["central", "extrapolate"] = "central",
    extrapolation_spacing: float | None = None,
    extrapolation_side_steps: int = 1,
    extrapolation_degree: int = 1,
) -> ForceConstants
```

`calculator` 必须是 ASE `Calculator`。`evaluate()` 只计算 central plan 的力并返回绑定计划的
`ForceBatch`；`run()` 串行完成计算与重建。`progress(done,total)` 在每次力计算后调用。

开启 ASR 时，重建先得到无约束物理 orbit 参数，再使用与力拟合相同的逐阶欧氏投影器。
`asr_tolerance` 是有限且为正的相对残差停止准则。重建日志会报告投影前后残差、参数修正和投影迭代次数。

`derivative_backend="extrapolate"` 会在多个正步长执行完整 central plan，并以 $h^2$ 多项式外推到零步长：

- `extrapolation_spacing`：相邻步长间隔，必须显式给出且为正；
- `extrapolation_side_steps`：基础步长两侧的层数，至少 1；
- `extrapolation_degree`：关于 $h^2$ 的拟合次数，至少 1；
- 所有生成步长必须保持为正，采样点数必须足以支持所选次数。

central 模式下提供任何非默认 extrapolation 参数会被拒绝，避免参数被静默忽略。

```python
calculation = FiniteDifferenceCalculation(
    primitive, order=2, reference=reference, cutoff=7.7237404951
)
fc2 = calculation.run(calculator, acoustic_sum_rule=True)
```

返回的 `ForceConstants.metadata` 记录 order、实际解析后的 cutoff、位移、空间群、ASR、
`asr_projection_tolerance`、构型数、导数后端，以及计划指纹 `plan_fingerprint` 与
`plan_schema_version`。

## 重建如何得到参数

每个 orbit 的 `observation_rows` 给出计划必须观测的分量行，重建显式求解
$Q_{\mathrm{obs}}\theta = y_{\mathrm{obs}}$，其中 $Q_{\mathrm{obs}}$ 是该 orbit 的观测矩阵，
$y_{\mathrm{obs}}$ 是有限差分得到的导数分量。求解结果 $\theta$ 是 $Q$ 的系数，随后由
`expand_primitive_parameters` 展开为带 site 与整数平移标签的力常数。它们不是参数 pivot：
观测分量一般不等于参数值，两者的差别由 $Q_{\mathrm{obs}}$ 的条件数控制（随 orbit 保存为
`observation_condition`）。

## 本轮破坏性变更

本分支处于开发阶段且不做兼容，迁移时请注意：

- `cutoff=None` 已删除，必须给出显式半径或负壳层；
- `sow()` 返回 `DisplacementBatch` 而不是裸列表；
- `evaluate()` 返回 `ForceBatch`，`reap()` 只接受 `ForceBatch`；
- 旧 `sow()` 输出没有指纹，无法再被收割，必须重新生成位移构型；
- 错误信息指向新 API，不提供兼容开关或静默回退。
