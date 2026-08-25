---
title: 力常数表示与 realization API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 力常数表示与 realization API

## `ForceConstants`

```python
ForceConstants(
    arrays: dict[int, np.ndarray],
    supercell: Atoms,
    metadata: dict[str, object] = {},
    sparse: dict[int, SparseOrderForceConstants] = {},
    relation: object | None = None,
)
```

普通用户不应手工构造该对象；有限差分、拟合、SCPH/SSCHA 或 `read_hdf5()` 会返回它。

- `sparse` 是 canonical primitive exact-$R$ Taylor IFC，是物理模型主体。
- `arrays` 是按某一 target materialize 后的缓存，不是 canonical 身份。
- `supercell` 是当前目标视图的 ASE `Atoms`。
- `relation` 是经过验证的 primitive/reference 对应，materialization、采样和外部 writer 需要它。
- `metadata` 保存来源、设置和约束记录，不参与 IFC 数值。
- `periodic_fc2_completion` 默认为 `None`；启用 periodic FC2 completion 拟合后，它保存仅属于训练
  reference 的有限周期谐波响应。该对象不改变 `sparse[2]` 的 exact-$R$ 身份。

### `orders`

```python
force_constants.orders -> tuple[int, ...]
```

返回 sparse 与 dense 缓存中所有可用阶的升序并集。

### `materialize()`

```python
materialize(
    order: int,
    *,
    max_bytes: int | None = 2_000_000_000,
) -> np.ndarray
```

把当前 target relation 上的 sparse IFC 稠密化。形状为

$$
(N_{\mathrm{primitive}},\underbrace{N_{\mathrm{reference}},\ldots,N_{\mathrm{reference}}}_{n-1},
\underbrace{3,\ldots,3}_{n}).
$$

`max_bytes` 是 warning 阈值，不是硬限制；超过阈值仍会继续申请内存。设为 `None` 可关闭 warning，但不减少
实际内存。高阶 IFC 应优先保持 sparse HDF5 或直接使用格式 writer。

## `SparseOrderForceConstants`

```python
from mlfcs.force_constants.representation import SparseOrderForceConstants

SparseOrderForceConstants(
    order: int,
    sites: np.ndarray,
    translations: np.ndarray,
    tensors: np.ndarray,
)
```

每行表示

$$
\Phi^{(n)}_{a_0a_1\ldots a_{n-1}}(R_1,\ldots,R_{n-1}),
$$

第一个原子锚定为 $(a_0,0)$。数组 shape：

- `sites`: `(K, order)` primitive site 标签；
- `translations`: `(K, order-1, 3)` exact primitive 整数平移；
- `tensors`: `(K, 3, ..., 3)`，含 `order` 个 Cartesian 轴。

## `realize_force_constants`

```python
realize_force_constants(
    force_constants: ForceConstants,
    reference: Atoms,
    *,
    primitive: Atoms | None = None,
) -> ForceConstants
```

将同一 primitive exact-$R$ IFC 映射到另一个合法目标超胞。`reference` 可与 source 大小、形状和原子顺序
不同，只要能与 primitive 建立整数超胞关系。`primitive=None` 时使用 source relation 的 primitive。

realization 对 folding 到同一 concrete cluster 的 exact interaction 求和，不重新拟合，也不改变 canonical
sparse rows。返回新 `ForceConstants`，其 `relation` 与 `supercell` 指向 target。

```python
target = build_supercell(primitive, (3, 3, 3))
target_fc = realize_force_constants(source_fc, target)
compact_fc2 = target_fc.materialize(2)
```

target 晶格、元素或 primitive 表示不等价时抛出 `ValueError`。该函数不能从小超胞观测“创造”source 中不存在
的 interaction，只负责展开已有 canonical IFC。

若对象包含 `periodic_fc2_completion`，只允许 realization 到具有相同整数平移商群的 reference（包括原子重排）；
不同大小或不同平移子晶格的 target 会被明确拒绝。`materialize(2)` 在合法 source 上返回 exact-$R$ FC2 与
periodic completion 的总和。
