# 本地测试

`tests/` 只包含确定性的单元测试和公共 API 回归测试。所有测试文件都扁平放置，并使用
`test_<area>_<behavior>.py` 命名。

本地运行：

```bash
uv run pytest
uv run ruff check src tests examples
uv run ruff format --check src tests examples
```

测试使用最小结构、固定随机种子，并根据单位或数值分析确定容差。测试不读取材料案例，不依赖外部软件，
也不使用开发者机器的绝对路径。

材料计算和第三方比较位于 `examples/`。案例 README 说明命令和结果，但不作为自动测试 oracle。
