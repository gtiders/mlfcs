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
uv run ruff check src tests reference_tools
uv run ruff format --check src tests reference_tools
uv run pytest -m "not reference"
uv build
```

仓库不再捆绑外部科学参考夹具。

## 测试要求

- 单元测试覆盖确定性的数学和 I/O 行为；
- 集成测试只使用公共 API；
- 科学结论必须提供独立参考、来源、单位、原子顺序映射、容差和独立 CI 步骤；
- 大型参考文件必须说明再分发条款并提供校验值；
- 不得把旧版 MLFCS 当作当前测试的真值。

拉取请求应保持范围清晰并解释科学或 API 动机。公共行为变化时同步更新中英文文档，
不要覆盖无关工作区修改，不提交构建产物，并在变更记录中说明用户可见变化。

贡献按照仓库的 GNU 通用公共许可证第 3 版或更高版本接收。
