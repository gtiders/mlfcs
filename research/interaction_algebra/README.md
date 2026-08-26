# Interaction algebra 统一生成元研究

## 结论

本研究实现了 primitive exact-$R$ interaction 与 finite pair 两种状态上的共享 indexed
generator traversal。结果为：

- primitive FC2-FC6 的 representative、image key、参数维数、pivot、normalized basis 和
  canonical-to-image action 与生产 exhaustive 实现一致；
- finite pair 的 symmetry/ASR 子空间以及 periodic FC2 exact complement 与生产实现一致；
- SymPy permutation group 的群阶与 MLFCS 仿射作用一致；
- 数学与架构结论为 GO，但当前 FC5/FC6 tensor invariant 原型不构成生产性能 GO。

FC6 使用 4.6 Å cutoff 和最大 3-body，仍有 44 个 orbit、4316 个参数和 62218 个 images。
完整 production baseline 为 65.23 s，优化后的 indexed generator 原型为 29.52 s，峰值 RSS
约 460 MiB。此前 7.70 s 的结果只覆盖 key traversal，不能代表完整 tensor orbit 构造。

优化的关键是将 tensor contraction 从每条 Schreier edge 执行一次，改为只对去重后的
stabilizer action 执行一次。FC6 的 435526 条 edge 最终只需 206 个 unique stabilizer actions；
FC5 从 50.11 s 降至 5.08 s，FC6 从 585.96 s 降至 29.52 s。

## 运行

快速验证 Si FC2-FC4、NaCl $2^3$/$3^3$ finite pair 和 periodic completion：

```bash
uv run python research/interaction_algebra/benchmark.py
```

加入 SnSe、Ba8Ga16Ge30 和 NaCl $4^3$：

```bash
uv run python research/interaction_algebra/benchmark.py --extended
```

加入 Si FC5/FC6 完整 tensor 验证：

```bash
uv run python research/interaction_algebra/benchmark.py --high-order
```

高阶命令可能占用数 GiB 内存并运行数分钟。机器摘要见 `results.json`。

## 实现边界

`indexed_orbit.py` 只处理固定宽度整数状态、生成元作用、复合 `TensorAction`、NumPy 排序索引
和 Schreier constraint Gram。它不依赖 `InteractionKey`、超胞或 FC2。

`primitive_prototype.py` 提供 exact-$R$ label codec、SymPy 空间群生成元、$S_n$ 相邻换位和
primitive orbit 装配。动态遍历不建立 `InteractionKey` hash；hash 只用于候选层缓存完整 orbit。

`finite_pair_prototype.py` 把有限 pair label 编为连续整数，使用布尔 visited 标记，并以同一内核
构造 sparse finite Hessian basis。

## Go / No-Go

- GO：共享 group-action/invariant 内核。
- GO：用生成元返回完整 canonical image action，而不只返回 key set。
- GO：finite pair 和 primitive orbit 共享内核与数值不变量构造。
- No-Go：将 source-periodic harmonic response 并入 transferable exact-$R$ interaction space。
- GO：indexed generator + 去重 stabilizer action 的 FC5/FC6 研究实现通过完整 tensor 对照。
- 暂缓：正式生产替换仍需把研究代码迁入 `src/mlfcs` 并增加回归测试。
- 暂缓：正式重命名 periodic FC2；先保持 source-owned force-constant 语义。
