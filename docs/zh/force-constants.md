# 力常数：模型、约束与文件

`ForceConstants` 是有限差分或拟合产生的原胞级模型。它将一个或多个阶次的参数块绑定到定义其物理含义的 `ClusterSpace`。

## 构造并查看模型

通常使用两种工作流的结果对象构造模型：

```python
fc_fd = fd.reconstruct(evaluated_structures)
fc_fit = system.force_constants(parameters)
```

直接构造入口为 `ForceConstants(space, coefficients)`。`coefficients` 将每个已包含的整数阶次映射到有限的参数向量，向量长度由该阶参数块决定。`orders`、`fingerprint`、`parameters(orders=None)` 和 `coefficients` 可查看模型内容。模型未包含的阶次不会自动视为零阶次。

同一空间中互不重叠的阶次可以合并：

```python
combined = ForceConstants.combine((fc2, fc3))
```

`save(file)` 保存原生 `.mlfcs` 表示，`ForceConstants.load(file)` 读取该文件。原生格式使用 pickle，只应加载可信来源的文件。

## 获取超胞张量

`model.get(order, mapping)` 返回未过滤的、以原胞原子为首轴的紧凑张量。形状为 `(n_primitive, n_supercell, ..., 3, 3, ...)`：首个原胞原子之后每个力常数指标对应一个超胞原子轴，每个张量指标对应一个笛卡尔轴。

```python
fc2_array = combined.get(2, mapping)
```

指定阶次必须存在于模型中，mapping 必须对应同一簇空间。`get()` 可访问模型包含的各阶；数组格式写出支持的范围小于张量访问范围。

## 声学和规则（ASR）

平移不变性要求整体均匀位移不改变原子受力。对应的声学和规则约束对所有平移像原子的力常数求和；高阶张量的每个相互作用指标都满足相应的平移约束。`ForceConstants.enforce_asr()` 将选定阶次投影到这些线性约束上。

```python
result = combined.enforce_asr(orders=(2, 3), rtol=1e-10)
model = result.force_constants
fc2_report = result.report(2)
print(fc2_report.relative_before, fc2_report.relative_after)
```

函数签名为 `enforce_asr(*, orders=None, rtol=1e-10)`。`orders=None` 选择模型中的全部阶次；显式给出时必须唯一且升序。`rtol` 是目标相对约束残差，必须为正数。方法返回新的 `ForceConstants` 模型和 `ASRResult`，不会原地修改输入。

每个 `ASRReport` 包含 `order`、`equations`、`parameters`、`residual_before`、`residual_after`、`relative_before`、`relative_after`、`correction_norm`、`relative_correction` 和 `iterations`。`ASRResult.report(order)` 可获取指定阶次的报告。

## 旋转不变性

旋转约束使无穷小旋转对应的力常数矩为零。API 作用于 FC2，并将一阶矩 Born-Huang 条件与二阶矩 Huang 条件分开。Huang 条件适用于无应力平衡结构，因此默认关闭。

```python
result = model.enforce_rotation(born_huang=True, huang=False)
model = result.force_constants
print(result.retained_rank, result.relative_before, result.relative_after)
```

函数签名为 `enforce_rotation(*, born_huang=True, huang=False, rank_rtol=None)`。`born_huang` 和 `huang` 至少选择一个。`rank_rtol=None` 时，根据观测到的几何残差推导奇异值秩截断；指定 `[0, 1]` 内的数值则直接设定相对截断。该参数决定实际笛卡尔几何下哪些旋转约束方向可解析。

`RotationResult` 返回投影后的 `force_constants`，并记录所选条件、`length_scale`、`equations`、声学/旋转残差的前后值、`relative_before`、`relative_after`、修正范数、`retained_rank`、秩截断及其自动/手动状态、保留/丢弃奇异值范围，以及几何和正交残差。未选择的 Born-Huang 或 Huang 条件，其报告字段为 `None`。

旋转投影不会施加 ASR，也不会改变输入模型原有的声学残差；它将旋转修正限制在声学零空间中。若要同时施加两类约束，先施加 ASR，再做旋转投影，这样旋转步骤会保留已满足的 ASR。返回报告可用于检查每一步的残差变化。

## 导出模型阶次

`model.write(file, mapping=None, *, format, order, threshold=1e-8)` 导出一个阶次。`order` 必须指定。`threshold` 是外部格式展开时使用的绝对分量截断值。对于需要目标超胞的格式，通过 `mapping` 指定。

| `format` | 支持阶次 | Mapping |
| --- | --- | --- |
| `phonopy_text` | FC2 | 必需 |
| `phonopy_hdf5` | FC2 | 必需 |
| `phono3py_hdf5` | FC3 | 必需 |
| `shengbte` | FC3 或 FC4 | 必需 |
| `tdep` | FC2、FC3 或 FC4 | 非必需；若提供，则会与模型核对 |

```python
model.write("fc2-phonopy.hdf5", mapping, format="phonopy_hdf5", order=2)
```

独立函数 `write_phonopy(file, array, mapping, *, format, threshold=1e-8)` 将已有 FC2 数组写为 `phonopy_text` 或 `phonopy_hdf5`。数组应与 `get(2, mapping)` 返回的原胞优先形状一致。

## 长程力

极性晶体可能具有缓慢衰减的偶极-偶极力常数。`Ewald` 计算器显式提供谐性长程分量；力的扣除和 FC2 重组流程见[长程力](long-range-forces.md)。[NaCl 教学案例](../../tutorial/NaCl/README.md)对比了直接拟合与显式长程修正。
