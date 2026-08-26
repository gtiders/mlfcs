# 数学库复用审计

## SymPy

采用 `sympy.combinatorics.PermutationGroup` 处理有限群阶、生成元闭包、Schreier-Sims 和
permutation representation 验证。primitive 仿射空间群先通过完整 operation composition table
映射为 regular permutation representation；finite pair 直接映射为 pair-index permutation。

未直接使用 SymPy `orbit()` 表示 exact-$R$ state。原因是 exact-$R$ label 含无界整数平移，
不是预先给定有限点集上的 permutation。先完整物化所有 exact-$R$ states 才能交给 SymPy，
会循环依赖于待求 orbit。该部分由 NumPy indexed traversal 完成，并由 SymPy 群阶及 exhaustive
image set 双重验证。

## NumPy

采用定长 `int64` rows、`np.unique(axis=0)`、排序后的 fixed-width row lookup、整数 state id 和
布尔 visited。复合 action 元数据保留为小型 rotation/permutation，不构造 Python key hash 图。

NumPy 还负责 exact affine label 变换、canonical row ordering 和 sparse builder 的索引装配。
canonical ordering 由领域 codec 显式提供；不能直接按交错 label row 排序，因为
`InteractionKey` 的顺序是先比较全部 sites，再比较 translations。

## SciPy

继续复用生产 `_null_space_from_gram()` 中的 `scipy.linalg.eigh()`、QR 子空间比较以及 LAPACK
线性求解。ASR null space 和 exact complement 对照沿用当前 SVD 实现。

rank-6 不构造 $729\times729$ Kronecker action；`TensorAction.apply_columns()` 通过逐轴 contraction
传播压缩 basis。尽管如此，FC6 的 Gram/eigensolver 和大量 rank-6 basis propagation 仍是主要
资源成本。当前进一步只对去重后的 stabilizer actions 执行 contraction，避免在每条 Schreier
edge 上重复传播相同的 tensor basis；后续仍可研究 block representation 或 symmetric
tensor-power basis。

## spglib 与 ASE

spglib 继续是晶体空间群操作的唯一来源，研究不重新识别空间群。ASE 继续负责结构容器、邻居
枚举和 supercell 构造。

## 保留的自定义逻辑

- spglib 仿射操作在 exact integer translation label 上的作用；这是 MLFCS canonical IFC 语义。
- `InteractionKey` 的 anchor/reanchor 与领域 canonical ordering。
- tensor axis permutation 与 Cartesian representation 的同步复合。
- dynamic exact-$R$ state 到连续整数 state id 的建立。
- finite pair $(a,b,[R])$ 与 reference atom index 的精确映射。

这些逻辑没有成熟库可直接提供，并已分别由 SymPy finite-group representation、生产 exhaustive
枚举和 periodic FC2 子空间对照验证。
