# 参与 MLFCS 开发

[English](CONTRIBUTING.md)

欢迎提交问题报告、参考数据、文档修正和范围清晰的代码改进。

## 提交问题前

请先搜索现有问题，并整理可复现的最小案例。报告中请包含 MLFCS、Python、ASE、NumPy、
SciPy 和 spglib 版本，操作系统，原胞结构和晶格，阶数、截断半径、超胞与位移设置，以及完整
traceback 或数值对比。未经许可，请勿附带不可再分发的势函数或计算数据。

## 开发环境

```bash
git clone https://github.com/gtiders/mlfcs.git
cd mlfcs
uv sync --group dev --group reference
```

运行数值测试和独立参考对比：

```bash
uv run pytest -m "not reference"
uv run pytest -m reference
uv run ruff check src tests
uv build
```

reference 依赖组为独立精确代数对比提供 SymPy。调用 phonopy 或 phono3py 的教学脚本使用可选的
tutorial 依赖组：

```bash
uv sync --group tutorial
```

文档贡献者可安装 docs 依赖组，并运行双语页面检查和严格构建：
`uv sync --group docs`、`uv run python docs/scripts/check_docs.py` 和
`uv run mkdocs build --strict -f mkdocs.yml`。

## 测试与教学案例

数值测试应聚焦可复现行为；精确代数和科学结果在适用时应与独立参考实现比较。不要为
`mlfcs.tools` 添加常驻测试；工具修改通过临时验证确认，验证代码不得提交。每个教学拟合任务由
自己的脚本捕获完整输出，并将覆盖后的 `fit.log` 保存在任务目录且纳入 Git。

教学计算可能需要另行准备结构、外部程序或大型参考数据。请记录这些输入及其来源；只有允许
再分发时才添加文件。

## Pull Request

保持修改范围清晰，并说明科学或 API 动机。公开行为变化时同步更新中英文文档。保留无关的工作树
修改，不提交构建产物；用户可见的变化应更新 changelog。本项目采用 GNU GPL v3.0 或更高版本。
