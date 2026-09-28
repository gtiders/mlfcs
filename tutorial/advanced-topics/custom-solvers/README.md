# 自选力常数拟合求解器

这个独立案例展示两种接入位置：保留原始设计矩阵后求解，或只保留 MLFCS 已构造的正规方程后求解。两条路线得到的都是同一 `ClusterSpace` 下的物理参数，最后分别交给 `FitData.force_constants()` 和 `FitSystem.force_constants()`，生成可保存、可继续用于后处理的 `ForceConstants`。

`fit.py` 在内存里生成一个两原子的教学晶胞和 24 个带有已保存 ASE 力的位移结构，因此不需要外部计算器或其他案例目录。截断半径 0.1 Å 只保留原子自身的二阶项；这是为了让求解器差异清楚可见，并非真实材料的截断建议。

从仓库根目录运行：

```bash
uv run python tutorial/advanced-topics/custom-solvers/fit.py
```

脚本每次覆盖 `fit.log`，并保存 `fc-direct.mlfcs`、`fc-normal.mlfcs`。它会检查两条路线的参数一致；如果秩不足或正规矩阵不是正定的，就明确失败。

## Gram 之前：保留原始方程

`FitData.from_atoms(mapping, structures)` 为每帧保留设计矩阵和力，`data.arrays()` 将它们堆叠为 $A$ 和 $f$。案例直接从 $A$ 计算每列范数，取 $D_{ii}=1/\lVert A_i\rVert$，用 `np.linalg.lstsq` 求 $(AD)z\simeq f$，再还原物理参数 $\theta=Dz$，交给 `data.force_constants(theta)` 构造模型。零范数列会明确报错；缩放不需要先构造 Gram，在多阶联合拟合时尤其有用。这里可以换成自己的 QR、SVD、LSQR、LSMR 或其他适用于原始方程的算法。代价是要存储随训练结构数增长的 $A$。

## Gram 之后：复用压缩系统

`FitSystem.from_atoms(mapping, structures)` 流式构造 $G=A^\mathsf{T}A$ 和 $b=A^\mathsf{T}f$，不保留所有单帧设计矩阵。如果已经构造了 `FitData`，也可以调用 `data.normal_system()`，无需再次遍历训练结构。案例先用 `system.column_scale` 对列作缩放，再用 SciPy 的 Cholesky 求解缩放后的 $G\theta=b$，最后还原物理参数并调用 `system.force_constants(theta)`。这里可以替换为自己的正定矩阵求解器；不同阶联合拟合时尤其不要忘记还原缩放。Cholesky 要求矩阵正定，且正规方程通常比原始最小二乘更敏感于病态数据。

这两条路径都不调用 `system.solve()`：该方法是 MLFCS 默认的列缩放 MINRES，而本案例专门演示如何替换求解步骤。换成实际训练数据时，保持结构是带有已保存力的 ASE `Atoms` 序列即可。
