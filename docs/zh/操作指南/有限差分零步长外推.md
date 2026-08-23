---
title: 有限差分零步长外推
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 有限差分零步长外推

[English] | 中文

外推后端在仍然只计算 `ForceConstantCalculation` 指定阶数的前提下，降低单一步长的
截断误差。它只用于 ASE Calculator 的直接串行执行：

```python
fc = calculation.run(
    calculator,
    derivative_backend="extrapolate",
    extrapolation_spacing=0.005,
    extrapolation_side_steps=2,
    extrapolation_degree=1,
)
```

构造函数中的 `displacement` 仍是中心步长 `h0`。后端生成：

```text
h(k) = h0 + k * extrapolation_spacing
k = -extrapolation_side_steps, ..., +extrapolation_side_steps
```

所有步长必须严格为正。每个步长都有一套完整中心差分子计划，因此 calculator 调用数
会乘以 `2 * extrapolation_side_steps + 1`。

中心差分误差按偶次幂展开，后端拟合：

```text
D(h) = D0 + c2 h^2 + c4 h^4 + ...
```

并取 `D0`。`extrapolation_degree=1` 是默认且通常最稳健的选择。阶数 `p` 保留到
`h^(2p)`，并要求位移步长数量大于 `p`。calculator force 含噪声时，更高拟合阶数不
一定更准确。

外推发生在对称性重建以及平移/旋转求和规则投影之前。程序报告相对中心步长的最大
修正、相对 L2 修正、多项式拟合残差和最终求和规则 drift。这些指标可用于发现位移
范围过窄、整体过大或已经被 force 噪声主导。

该后端有意不加入 `sow()` 和 `reap()`：外部流程继续对应一套确定性的位移计划，避免
原子顺序约定和外部计算目录数量变得含糊。
