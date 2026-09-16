---
alwaysApply: true
description: 重构期间的写操作只允许发生在 worktree .worktrees/reflector 内
---

# 写入边界：只在 `.worktrees/reflector` 内改代码

本项目正在重构。重构期间唯一允许产生文件改动的目录是 worktree：

```
/home/gwins/codespace/mlfcs-new/.worktrees/reflector
```

## 硬性要求

- `write`、`edit`、`ast_edit` 以及任何会生成文件的脚本，目标 MUST 位于上述 worktree 内，路径 MUST 写成 worktree 内的绝对路径。
- `bash` MUST 设 `cwd=/home/gwins/codespace/mlfcs-new/.worktrees/reflector`。安装依赖、建 venv、跑测试、格式化、生成产物一律在 worktree 内发生。
- Git 写操作（`add`/`commit`/`checkout`/`restore`/`stash`/`branch`/`merge`/`reset`/`rebase`）MUST 只作用于 worktree 内的仓库，分支以 `reflector` 为准。
- 主仓库工作树 `/home/gwins/codespace/mlfcs-new`（分支 `dev`）MUST NOT 被写入：不得新建、修改、删除其任何文件，也不得改动其索引、HEAD 或工作树状态。它带有用户尚未提交的改动，任何触碰都会污染用户的工作。
- `git worktree` 的查询类命令（`list`、`status`、`log`、`diff`）在主仓库目录里只读执行是允许的。
- 读取不受限：主仓库的 `src/`、`tests/`、`docs/`、`tutorial/`、`research/` 都可以读来做参考。

## 唯一例外

用户在当前会话里点名要求改动 worktree 之外的具体文件时才执行；其余情况一律先停下来说明原因，不要动手。

## 自查信号

- 目标路径里没有 `.worktrees/reflector`
- 主仓库的 `git status` 相比会话开始时多出条目
- 在 worktree 之外的目录里跑出 `uv sync`、`pytest`、脚本产物、`__pycache__`
