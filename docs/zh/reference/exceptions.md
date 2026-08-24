---
title: 异常与排错
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 异常与排错

当前公共工作流主要使用 Python 标准异常类型，消息中包含被拒绝的条件。MLFCS 不把失败降级为 warning。

| 异常 | 常见原因 | 首先检查 |
|---|---|---|
| `TypeError` | 不是 ASE `Atoms`/`Calculator`/`ForceConstants` | 对象类型与导入来源 |
| `ValueError` | 参数范围、结构关系、shape、order 或格式不合法 | 完整消息、reference 与单位 |
| `KeyError` | 请求不存在的 IFC order 或温度 | `force_constants.orders`、温度序列 |
| `RuntimeError` | 数值秩、JAX 设备、正规形或迭代内部保证失败 | 同一调用日志和秩信息 |
| `MemoryError` | 目标稠密张量超过可用内存 | 保持 sparse HDF5，避免高阶 materialization |
| `AlamodeMirrorImageError` | 目标超胞不能表达 ALAMODE 27-image 编码 | 更换目标超胞或输出格式 |

拟合、有限差分和 harmonic sampling 都把 reference 原子顺序视为权威顺序。若输入只是排列不同，先显式
使用 `align_structures()`；不要期望计算 API 自动重排。若晶格或 primitive 对应不同，必须修正输入结构。

当多个 primitive exact-$R$ interaction 在 reference 中折叠为同一个有限观测且导致 realization map
秩亏时，构造会拒绝继续。增加训练帧不能修复 representation kernel；需要更大的 reference 或更小 cutoff。

拟合默认拒绝未收敛求解；`allow_unconverged=True` 才会带 warning 返回最后参数。SCPH/SSCHA 达到最大
步数时会返回最后迭代并记录 `converged=False`，用户必须检查历史，不能只看是否生成了文件。
