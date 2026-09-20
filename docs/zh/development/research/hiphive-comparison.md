---
title: "与 hiPhive 的对比：初始化与表示"
audience:
  - advanced
  - developer
status: research
code_verified: 4.0.0a6
---

# 与 hiPhive 的对比：初始化与表示

本笔记比较**拟合或有限差分之前**的整个阶段——结构、对称性、簇发现、不变量基、可实现性——以及拟合随后使用的表示，
对照的是 MLFCS 4.0.0a6 与 hiPhive 1.5。hiPhive 一侧的陈述来自为本笔记阅读其源码，以下每个引用都是该发行版的文件与行号。

## 两者共有的部分

两边都把对称性代数放在晶格（scaled）坐标系中，这是决定性的共同选择：

- spglib 的旋转在那里是整数矩阵。hiPhive 通过 `cluster_space.rotation_matrices` 公布（`cluster_space.py:164`），
  MLFCS 通过 `PrimitiveSymmetryOperations.rotations`；
- 第一层基都是 label 对称指示基：`init_ets_from_label_symmetry`（`core/eigentensors.py:47`）与
  `label_symmetric_basis` 是同样的 $0/1$ 构造；
- 参数化都是晶格框架下的基，笛卡尔分量只出现在边界：hiPhive 在装配超胞力常数时经
  `rotation_to_cart_coord(R, cell)`（`core/tensors.py:38`）转换，MLFCS 在消费者需要物理张量时经一次确定性映射
  $K = (\text{cell}^T)^{\otimes\text{order}}$ 转换。两个公式是同一个映射，这也是
  [orbit-search-thresholds.md](orbit-search-thresholds.md) 中的框架标定能够与 spglib 自带笛卡尔旋转逐生成元对照的原因。

## 初始化阶段的差异

| 环节 | hiPhive 1.5 | MLFCS 4.0.0a6 |
| --- | --- | --- |
| 簇发现 | 先枚举超胞内**全部簇**，再做归约：`get_orbits(cluster_list, atom_list, rotation_matrices, ...)`（`core/orbits.py:18`）拿到的是已经建好的簇表 | 生成元闭包式 orbit 搜索：`iter_primitive_candidates` 给出的种子在生成元下闭合，配 Schreier 稳定子与规范锚点（`traverse_indexed_orbit`），从不整体枚举簇 |
| 不变量基 | 逐对称操作迭代约化：稀疏约束矩阵、`SparseMatrix.rref_sparse` / `nullspace` 的精确有理零空间（`core/utilities.py:25`），再用 `renormalize_to_integer` 整化（`core/eigentensors.py:112`） | 约束只取稳定子集合（Schreier 生成元，少于全群）；`int64` 整数 Gram；维数取大素数模秩；核用浮点解并由**整数逐列验证**，最后按列 gcd 归一 |
| 秩与精确性 | 由符号消元决定零空间 | 秩的**两个方向**都由素数证明（见 `certified_rank`）：模秩只会低估 $\mathbb{Q}$ 上的秩，故满秩即证明；判亏秩时持续加入素数，直到乘积超过最大子式的精确 Hadamard 上界 |
| 参数化 | eigentensor **就是**参数，没有 pivot 概念 | 整数晶格基即参数化，拟合参数是它的系数；另有 `pivots`——有限差分计划观测、且能确定这些系数的笛卡尔分量行 |
| 参考胞可辨识性 | 没有等价检查：欠定参考由 ridge 正则化吸收（`enforce_rotational_sum_rules` 用 `Ridge(alpha=...)`） | `validate_realization_identifiability` 对整数晶格实现矩阵按连通分量求精确秩，秩亏即抛 `InteractionAliasingError`，并指出冲突团簇与两条补救办法 |
| 求和规则 | 旋转和规则与声子求和规则都是**拟合后**投影 | 声子求和规则作为等式约束进入拟合（带约束最小二乘）；旋转和规则仍在拟合后处理 |
| 截断 | 调用方显式给距离 | 调用方显式给距离或负壳层序号（"由参考胞解析"的 `cutoff=None` 已删除，两个 API 在风格上因此一致） |
| 失败模式 | 折叠参考胞给出被正则化压扁的解 | 折叠参考胞在实现轨道空间时被拒绝 |

实质后果就体现在最后一行：hiPhive 下参考胞不足是**静默**的（正则化把它吸收掉），MLFCS 下是报错并指出需要修的团簇——
正因如此，我们才删掉了"由参考胞解析截断"的模式，把职责交给这项检查。

## 实测代价

四阶不变量核，每轨道，本机实测：

| 路线 | 代价 |
| --- | --- |
| 照搬 hiPhive 的配方（其 `SparseMatrix` 零空间，按点群迭代） | 30–70 ms |
| MLFCS（`invariant_kernel`：整数 Gram、模秩、验证过的核） | 0.4–2 ms |

为 Si 的四阶空间全部 12 个 Gram 出精确秩证书耗 361 ms，两素数快路径 42 ms，被拒绝的参考胞 1–48 ms 内报出。
差距来自"每个对称操作做什么"：带系数增长的符号消元，对比其判定要么是素数下的证明、要么在整数上被验证的整数运算。

## 已提交的 NaCl 案例数值对照

长程静电力教程在同一份上游数据上跑两个代码（8 原子 rocksalt 胞、$4\times4\times4$ 参考胞、两个有限位移构型、
11 Å 的 FC2 截断）。库内已提交的指标：

| 量 | MLFCS | hiPhive |
| --- | --- | --- |
| 参数个数（MLFCS 为 23 个 orbit） | 76 | 74 |
| 短程拟合力 RMSE（eV/Å） | $1.484222611000188\times10^{-5}$ | $1.4842226109978366\times10^{-5}$ |
| 总力拟合 RMSE（eV/Å） | $2.651939354365933\times10^{-5}$ | $2.6519393543682293\times10^{-5}$ |
| 相对力误差（短程） | $0.024691623576418457$ | $0.024691623576379346$ |
| 恢复长程后的 FC2，相对 Frobenius 差 | $4.27\times10^{-10}$ | -- |
| 声子带相对 phonopy+NAC 的差异（短程拟合，THz，最大） | $0.11648212566859484$ | $0.1164821254595414$ |

复现方式：该教程目录下的 `prepare.py`、`fit_mlfcs.py`、`run_hiphive.py`；hiPhive 1.5 因 numba 目前要求
NumPy 低于 2.5，故其命令会建一个临时环境。

仍需解释的差异是参数个数 76 对 74，而拟合出的 FC2 相对差 $4.3\times10^{-10}$：两套参数化在这份数据上等价，
但不是同一组数字。

## 有意没有从 hiPhive 取用的部分

- 符号路线：精确，但四阶每轨道 30–70 ms（本仓库的四阶空间合 13–44 s），而字母表前整个不变量阶段只要 0.096 s；
- 用正则化处理欠定参考胞：它给出答案而不是错误，而参考胞恰恰是调用方能够修的东西；
- 其簇枚举顺序：成本随超胞内簇数增长，而非随 orbit 大小增长。
