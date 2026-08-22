---
title: Contributing
audience:
  - developer
status: stable
code_verified: 4.0.0a4
---

# Contributing

Keep public behavior documented by a runnable example or a focused unit test. New user-facing
pages require an English and Chinese counterpart, a navigation entry in both MkDocs configs, and
one command that can be checked without a proprietary calculator.

Run the local checks:

```bash
uv run ruff check src tests examples
uv run pytest -m "not reference"
uv run --group docs mkdocs build --strict -f mkdocs.yml
uv run --group docs mkdocs build --strict -f mkdocs.yml
```
