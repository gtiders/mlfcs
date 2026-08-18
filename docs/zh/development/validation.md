# 数值验证

数值验证分为确定性的本地测试和材料案例。本地测试检查数学恒等式、结构映射、稀疏支撑、writer 契约和
小型公共工作流，不由 CI 运行。

解析 Morse FC4 测试使用独立能量表达式，验证有限差分阶数和 FC4 数值，不依赖另一套力常数实现。

过去位于 `tests/reference/` 的材料比较现在归入对应的 `examples/<Material>/` README。第三方输出和
来源作为人工审查的证据保留，不再作为自动真值数组。比较 IFC、声子或输运结果前，README 必须说明
primitive、supercell、原子顺序、单位、cutoff、周期镜像约定和约束。

可信度应综合力误差、不变量、对齐后的 IFC/动力学矩阵、声子稳定性与收敛、NAC 设置以及输运收敛。
上游结构或 q 空间约定不一致时，即使观测量接近，也不能据此证明 writer 正确。

## 本地检查

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

## 文档 CI

CI 只使用严格模式构建中英文文档站并检查链接和导航，不运行 Python 测试、第三方 calculator、科学基准或
Python 包构建。
