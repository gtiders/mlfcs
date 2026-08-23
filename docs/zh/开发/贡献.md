---
title: 贡献
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# 贡献

公共行为必须由可运行案例或聚焦单元测试说明。新增用户页面必须提供对应的中英文页面、在两个 MkDocs
配置中加入导航，并提供一个不依赖专有 calculator 的可检查命令。

本地检查：

```bash
uv run ruff check src tests examples
uv run pytest -m "not reference"
uv run --group docs mkdocs build --strict -f mkdocs.yml
uv run --group docs mkdocs build --strict -f mkdocs.yml
```
