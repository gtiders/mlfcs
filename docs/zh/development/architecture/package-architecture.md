---
title: MLFCS 4.0 技术概览
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# MLFCS 4.0 技术概览

[English] | 中文

## 范围与架构

MLFCS 基础流程以 ASE 为公共边界，根据用户提供的力重建力常数，不内置势函数。`ForceConstantCalculation` 用同一套 `order` 参数化算法处理
二阶及更高阶；`structure` 负责整数格、结构关系和周期几何，`interactions` 负责团簇、
轨道与有限超胞 realization，`basis` 负责 Wick/Taylor 变换，`force_constants` 负责规范
力常数表示，`constraints` 负责物理约束，`io` 负责格式适配。
原生 `mlfcs.physics.sscha` 直接使用
compact FC2 的相容 q 点采样和同一套 Gram 拟合参数化。

源码按职责组织为：

```text
structure → interactions → basis
structure + interactions → force_constants
底层表示 → constraints / finite_difference / fitting
force_constants + fitting → physics
structure + force_constants → io
```

这里的箭头表示允许的 import 方向，不是强制工作流。`physics` 和 `io` 是终端层，底层
数据对象不会反向导入 writer 或高层求解流程。

## 超胞、截断与顺序

计算使用显式 `StructureRelation`，把 primitive 与用户顺序的 reference 超胞关联起来；
一般整数 `3×3` 超胞矩阵和原子任意重排均受支持。基于 HNF 的 `PeriodicIndex` 只负责把
`(primitive_site, exact_translation)` realization 到当前 reference，避免依赖浮点坐标反推
或隐藏的 cell-major 顺序。负整数截断表示近邻壳层；程序分别报告 primitive 晶格能够枚举的
最大壳层/半径，以及用户实际选择的壳层/半径。`sow()` 的零基构型 ID、
原子顺序和位移数组会随结构保存，位置式 `reap()` 严格遵循写出顺序，字典式 `reap()`
则按构型 ID 接收乱序结果。

## 通用高阶流程

空间群操作、力常数指标置换、笛卡尔旋转和稳定子约束共同把团簇张量约化为独立参数。
团簇发现不再遍历全部邻居标签的有序笛卡尔积，而是递归生成非递减的邻居多重集合；
任意新增原子违反 all-pair 截断时，整个无效前缀立即停止扩展。空间群在 primitive site 与
exact 整数平移上作仿射变换，随后重新锚定，使第一个平移为零。重复原子对应的笛卡尔轴先进入标签
对称基，再累积稳定子约束，并以矩阵无关的 NumPy 张量收缩避免生成完整的
`3**order × 3**order` 作用矩阵。该流程保持 FC2--FC4 轨道子空间，同时显著降低重复
原子 FC5/FC6 的内存需求。
`n` 阶力常数由力的 `n-1` 阶中心差分产生，符号组合递归生成并在计算力之前去重。
重建结果以 primitive sites、exact 整数平移和局部 `3 × ... × 3` 张量保存，只有显式指定
目标超胞时才物化完整稠密
数组。因此二阶至任意高阶共享端到端流程；三阶和四阶是主要生产验证路径，五阶已经
完成烟雾验证，更高阶受组合规模限制并应使用通用稀疏 HDF5 导出。

ASE Calculator 直接运行时，可选外推后端围绕设定位移构造多套完整中心差分子计划，
分别收缩导数并按照 `h^2` 的多项式拟合。零步长截距随后进入同一套重建与求和规则
流程；外部 `sow()` / `reap()` 仍保持单一确定性计划。

## ASR 与数值方法

本次模块整理的硬性要求是保持当前数值实现：不改变算法、默认值、约束方程、稀疏
标签、HDF5 schema 或 writer 语义。旧的兼容转发模块已经删除；新代码使用
`mlfcs` 或职责对应的子包。所有用户入口在定义处直接声明完整签名，拟合和 SSCHA
按需加载，不会因为导入有限差分或 IO 而初始化 JAX。

平移不变性要求固定其他指标时，对任意一个原子指标求和为零。MLFCS 在不可约轨道参数
空间构造约束 `A p = 0`，并对所有规模统一使用稀疏、矩阵无关的 LSMR 最小范数投影，
不构造 `A.T @ A`，从而避免 Gram 存储的二次增长和条件数平方问题。这与
旧版相对权重补偿不同：后者不保证严格满足约束，旧四阶还曾错误地同时对两个原子轴
求和。

有限差分与拟合共享相互作用空间、平移约束构造、残差定义和最终 LSMR 投影，但顶层
求解保持分离：有限差分投影已经重建的参数，拟合则在约束零空间内最小化力残差。当前
拟合零空间投影仍形成稠密 `C @ C.T`；大约束系统的自动稀疏替换已列入路线图。

## CPU、GPU 与内存优化

JAX 只用于联合拟合中的稠密 Wick 力特征核；团簇枚举、张量作用、有限差分模板、稀疏
约束、重建和 I/O 都保持 NumPy/SciPy 宿主端实现。`jax_platform` 可选 `auto`、`cpu` 或
`gpu`，它选择拟合使用的显式设备，而不改写 JAX 的进程全局后端。每次拟合只准备一次
静态相互作用缓冲区和已编译核；GPU 的物理设计、分块稀疏约化与 Gram 累积全程驻留设备。
连续稀疏数组、位移去重、串行 calculator 计算和惰性稠密物化共同控制内存。超过建议
预算时程序发出警告而非强制拒绝，通用稀疏 HDF5 不需要完整稠密数组。

## I/O 与兼容性

`format` 显式选择输出：任意阶可用原生 `hdf5`，二阶可用 phonopy 文本或 HDF5，
三阶可用 phono3py HDF5，三阶和四阶可用 ShengBTE。ShengBTE 直接写出 canonical IFC 的
exact primitive 平移，不重新搜索周期像。

`alamode` 可在同一个 FCSXML 中写出 FC2--FC4。适配器保留 reference 超胞顺序，并直接序列化
已有的 `primitive_index`/`cell_translation` 映射，而不重新识别原胞。周期镜像编号遵循
ALAMODE 固定的 27 像约定；导出会先尝试等价的 Minkowski 约化超胞换基，仍无法表示真实
最小镜像时才明确拒绝。

新版应在超胞几何、近邻壳层、原始重建、原子映射和格式语义上与参考实现核对，但不会
模拟物理约束错误的旧 ASR。详细数值见[验证文档]。

## 当前边界

组合数量会随 `order!`、`3**order` 和 `2**(order-1)` 快速增长。基础流程尚不包含
非解析长程静电修正、FC3 bubble 或 FC4 loop 自能；原生 SSCHA 通过热力采样得到有效
FC2，但不是显式图形自能计算。
