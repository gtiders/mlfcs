---
title: 日志
audience:
  - user
status: stable
code_verified: 4.0.0a5
---

# 日志

MLFCS 通过名为 `mlfcs` 的标准 Python logger 报告主要物理状态和数值状态。默认情况下，
`INFO` 及以上级别全部写入 stdout。软件不配置 root logger，也不把 Python 未捕获异常的
traceback 从 stderr 重定向出去。

如需查看 batch、计时和秩判据等实现细节，可启用 `DEBUG`：

```python
import logging

logging.getLogger("mlfcs").setLevel(logging.DEBUG)
```

用户可以使用 Python 标准 logging API 添加过滤器或替换 handler。MLFCS 公共工作流不再
接受 `verbose`、`log_level` 或 reporter callback。非法状态直接抛出异常；warning 只表示
计算仍然合法但其后果需要注意，例如显式选择虚频处理策略、位移被裁剪或返回未收敛迭代。
