---
title: 日志
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 日志

MLFCS 使用标准 Python logger `mlfcs`。导入包时安装唯一 stdout handler，默认显示 `INFO` 及以上；
不会调用 `logging.basicConfig()`、不会修改 root logger，也不会吞掉 Python traceback。

```python
import logging

logging.getLogger("mlfcs").setLevel(logging.DEBUG)
```

`DEBUG` 用于 batch、缓存、计时和秩阈值等细节。公共 API 不提供 `verbose`、`log_level` 或 reporter
参数；进度回调只存在于必须逐构型计算的有限差分和 SSCHA `run()` 中。

若案例必须保存完整日志，可给 `mlfcs` logger 增加文件 handler，同时捕获计算器写到 stdout/stderr 的内容。
不要依赖终端重定向去捕获已在导入时绑定的 package handler。

- `INFO`：物理设置、orbit/参数数、拟合误差、迭代状态和导出目标。
- `WARNING`：仍返回合法结果但需要关注，例如显式接纳虚频、位移裁剪或允许未收敛结果。
- 异常：非法输入或不能保证物理语义时直接抛出，不用 `ERROR` 日志代替。
