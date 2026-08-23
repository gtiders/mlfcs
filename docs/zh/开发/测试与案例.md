---
title: 测试与案例
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# 测试与案例

MLFCS 有两个彼此独立的本地验证入口：

| 位置 | 责任 | CI |
|---|---|---|
| `tests/` | 确定性的单元测试和公共 API 回归 | 不在 CI 运行 |
| `examples/` | 材料工作流和第三方参考证据 | 不在 CI 运行 |

## 测试

所有测试文件都扁平放置，并使用 `test_<area>_<behavior>.py` 命名。测试使用最小结构、固定随机种子和
有单位依据的容差；不读取案例、不依赖外部软件，也不使用开发者路径。

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

解析 Morse FC4 测试是内部数学 oracle，不是第三方材料基准。材料比较和热导率结果不会作为普通 pytest
真值数组。

## 示例与案例

顶层 `examples/*.py` 演示单个公共 API 任务。材料数据统一位于：

```text
examples/<Material>/<case>/
  README.md
  structures/
  fitting/
  finite_difference/
  results/
  observables/
```

每个 README 记录结构、原子顺序、单位、cutoff、calculator/软件版本、命令、输出角色、下游 q 网格和已知
差异。第三方结果保留原格式并作为参考证据；案例不要求统一 manifest、checksum 或 check 脚本。

拟合数据使用严格的 ASE `extxyz`；有限差分数据保留有序的 sow 工作区和 `mlfcs-plan.json`。两种工作流都以
参考原子顺序为权威。

文档 CI 只构建中英文文档站。材料案例在重新生成或审查结果时手动运行。
