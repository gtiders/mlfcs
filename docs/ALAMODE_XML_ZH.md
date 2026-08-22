# ALAMODE XML 输出

[English](ALAMODE_XML.md)

MLFCS 可以把二阶至四阶力常数写入同一个 ALAMODE FCSXML 文档：

```python
force_constants.write("force_constants.xml", format="alamode")
force_constants.write("fc3.xml", format="alamode", order=3)
```

省略 `order` 时，会写出当前结果中所有可用的 FC2、FC3 和 FC4。更高阶仍可使用 MLFCS
HDF5；这里实现的 ALAMODE FCSXML 模式只定义了 `HARMONIC`、`ANHARM3` 和
`ANHARM4` 区段。

## 原子顺序与原胞映射

导出的 `Structure/Position` 顺序与 `force_constants.supercell` 完全一致，不会按元素、
原胞原子或晶格平移排序。随后由 MLFCS 显式写出映射：

- `primitive_index[i]` 指定超胞原子 `i` 对应的原胞原子；
- `cell_translation[i]` 指定该原子的整数超胞平移；
- `Symmetry/Translations/map` 记录原胞原子和平移到超胞序号的完整查找表。

因此，用户可以通过所需顺序构造 MLFCS 超胞，并保持上述两个元数据数组一致，从而控制
XML 原子顺序。writer 会验证映射完整且一一对应，不会再次运行 spglib，也不会静默修改
映射。

FCSXML 力常数 pair 遵循 ALAMODE 约定：第一个原子使用原胞原子编号，其余原子使用准确
的超胞编号，最后一个整数是周期镜像晶胞编号。这只是文件格式表示，不是额外的原子重排。

## 周期镜像与单位

ALAMODE 使用固定的 27 个镜像晶胞编号：中心胞之后是 `{-1, 0, 1}^3` 中的全部平移。
存在多个等距最近镜像时，MLFCS 会分别写出，并按镜像组合数平分力常数，与 ALM 官方
Python writer 一致；同一原子在一个团簇中重复出现时共享同一个镜像选择。MLFCS 还会
用 ASE 的通用最小镜像进行核验。若固定 27 像无法表达真实最小镜像，程序会先自动对目标
超胞作整数幺模 Minkowski 换基；该换基保持物理超胞晶格、原子和 IFC 不变，通常可让非约化
表示落入 27 像范围。换基后仍无法表达时才拒绝导出，避免写出错误的周期几何。

MLFCS 中 `n` 阶力常数单位为 eV/angstrom^`n`。FCSXML 使用
`value * bohr^n / Ry` 换算为 Ry/bohr^`n`；晶格矢量写为 bohr，分数坐标不变。

## 来源与兼容性说明

XML 布局和镜像展开改编自 `ttadano/ALM` 仓库
`f1d668f210d3e95355643132144f3fd1ec10d4d7` 版本中采用 MIT 许可证的
`alm.fcsxml.Fcsxml` writer；完整来源和许可条款嵌入在 `src/mlfcs/io/alamode.py` 顶部。语义测试覆盖
多组分原子顺序、平移映射、镜像简并、重复原子、单位换算和 FC2--FC4 区段。

ALAMODE 的另一个 Python XML reader 会自行调用 spglib；它在某些晶体中可能选择平移过
原点的原胞，并拒绝本来有效的 XML，甚至 ALM 自己的 Python writer 输出也可能触发该
限制。MLFCS 不模仿这种依赖原点的重新识别，XML 中的显式映射才是下游 ALAMODE 计算
所使用的权威关系。
