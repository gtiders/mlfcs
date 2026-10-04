# 日志

MLFCS 导入时安装一个包日志 handler，默认 INFO。每行固定包含本地时间、
级别、模块和消息：

```text
2026-10-03 14:15:00 INFO mlfcs.fitting.system: Fit solve started: ...
```

日志写入并刷新当前 Python stdout，文件及重定向由调用者管理。
配置只提供级别，不提供 `stream`、文件路径或自定义格式：

```python
import logging
from mlfcs.core.log import configure

configure(level=logging.DEBUG)    # 详细诊断
configure(level=logging.WARNING)  # 只显示警告和错误
```

重复配置只修改级别，不重复安装包 handler，不修改 root logger 或用户添加的 handler。

## 保存任务输出

用 Bash 将包日志、普通打印、stderr 和未捕获异常 traceback 覆盖保存到同一文件：

```bash
python -u run.py > run.log 2>&1
```

边显示边保存，并保留 Python 的失败退出状态：

```bash
set -o pipefail
python -u run.py 2>&1 | tee run.log
```

教学拟合有单独要求：每个任务脚本自己覆盖本目录的固定 `fit.log`，完整捕获
stdout、stderr 和 traceback。日志作为案例结果纳入 Git，各脚本自行管理捕获，
不依赖共享包装器。包 handler 跟随脚本当前的 stdout，包括文件关闭后恢复的 stdout。

## 计算摘要

INFO 覆盖 cluster 构造、对称性/candidate/orbit、超胞映射与 folded rank、
ForceDesign 编译、拟合、有限差分、谐波网格、SCPH、和规则投影及模型保存/加载/导出。
成功完成时记录有关规模、结果和耗时。拟合及计算器评估在第一个项目和每十个项目后
记录进度；拟合迭代器只消费一次。耗时包含可能发生的 JIT 编译。

求解日志给出算法、停止设置、终止码和迭代次数；拟合摘要给出物理尺度的训练力 RMSE
（eV/angstrom）及相对力误差。正规方程从已保存的充分统计量计算诊断，
不会恢复已丢弃的逐帧力。若浮点消去导致诊断不可用，只记录警告，不使成功求解失败。
INFO 的质量摘要额外计算一次残差；WARNING 级别关闭这项工作。日志不修改拟合系数。

DEBUG 增加 fingerprint 和 workspace 细节。昂贵的 fingerprint 遍历受 DEBUG
级别保护，INFO 不执行这些日志专用遍历。

## 异常

科学 API 保持正常抛出异常，不安装全局异常 hook，也不在每层重复记录同一异常。
任务边界需要记录已捕获异常时使用标准接口：

```python
from mlfcs.core.log import get_logger

logger = get_logger("task")
try:
    main()
except Exception:
    logger.exception("Task failed")
    raise
```

需要单份 traceback 时，选择任务边界记录或未捕获 traceback 捕获其一。
异常对象无需自定义 `.log()` 方法。
