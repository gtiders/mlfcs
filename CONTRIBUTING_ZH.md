# 参与 MLFCS 开发

[English](CONTRIBUTING.md)

欢迎提交缺陷报告、独立参考数据、文档修复和范围明确的拉取请求。

## 提交 issue 前

请先搜索已有 issue，并提供可复现案例，包括：

- MLFCS、Python、ASE、NumPy 和 JAX 版本；
- 操作系统和 CPU/GPU 后端；
- 原胞、超胞、阶数、截断、位移量和 ASR 设置；
- 完整报错或数值比较；
- 力来自 `run()` 还是外部 `sow()` / `reap()`。

未经许可不得上传专有势函数或计算数据。

## 开发环境

```bash
git clone https://github.com/gtiders/mlfcs.git
cd mlfcs
uv sync --locked --dev
```

提交前应通过：

```bash
uv run ruff check src tests reference_tools examples
uv run ruff format --check src tests reference_tools examples
uv run pytest -m "not reference"
uv build
```

科学参考必须串行执行，并可能耗时较长。pypolymlp 比较还需要 Eigen 头文件及专用依赖组：

```bash
uv sync --locked --dev --group reference
uv run pytest tests/reference/analytic/Morse_FCC_FC4/test_morse_fc4.py
```

本地只运行与改动相关的参考测试，完整序列由 CI 执行。

## 测试要求

新增或迁移验证内容前必须遵循[测试与案例设计原则](docs/TESTS_AND_EXAMPLES_ZH.md)。
外部软件的材料数值对比属于 `examples/cases`，不应新增为普通 pytest oracle。

- 单元测试覆盖确定性的数学和 I/O 行为；
- 集成测试只使用公共 API；
- 科学结论必须提供独立参考、来源、单位、原子顺序映射、容差和独立 CI 步骤；
- 第三方参考文件必须在案例 README 中说明来源和再分发条款；
- 不得把旧版 MLFCS 当作当前测试的真值。

拉取请求应保持范围清晰并解释科学或 API 动机。公共行为变化时同步更新中英文文档，
不要覆盖无关工作区修改，不提交构建产物，并在变更记录中说明用户可见变化。

贡献按照仓库的 GNU 通用公共许可证第 3 版或更高版本接收。
