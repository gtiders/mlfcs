---
title: API 参考
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# API 参考

这里记录 MLFCS 公开 Python 接口的完整调用契约。正文按“构造输入 → 计算 IFC → 处理与导出 →
有限温度工作流”组织，而不是照抄源码目录。示例默认使用：

```python
from ase.io import read
import mlfcs
```

## 顶层稳定入口

| 分类 | 顶层名称 | 作用 |
|---|---|---|
| 结构 | `build_supercell` | 按 phonopy old-style 顺序生成显式超胞 |
| 采样 | `perturb_structures` | 生成 Gaussian 或谐波分布位移结构 |
| 有限差分 | `FiniteDifferenceCalculation` | 构造、执行或回收对称约化有限差分 |
| 拟合 | `ForceConstantFitter` | 由单一 reference 的力快照拟合连续阶 IFC |
| 数据 | `ForceConstants` | 保存 canonical sparse Taylor IFC 与目标 realization |
| ASE 势 | `MLFCSCalculator` | 从 Taylor IFC 计算 reference-relative 能量与原子力 |
| 约束 | `enforce_rotational_sum_rules` | 对 FC2 施加 Born–Huang/Huang 修正并保持 ASR |
| 有限温度 | `LoopSCPH`、`SSCHA` | 生成温度相关有效 FC2 |
| realization | `realize_force_constants` | 将 primitive exact-$R$ IFC 展开到合法目标超胞 |
| I/O | `read_hdf5`、`write_force_constants` | 读取原生 HDF5，写出原生或外部格式 |

`mlfcs.__version__` 返回当前版本。`ForceConstantFitter` 和 `SSCHA` 延迟加载，因此
`import mlfcs` 不会主动初始化 JAX。

## 高级但有用的子模块对象

```python
from mlfcs.fitting import FittingResult
from mlfcs.force_constants.representation import SparseOrderForceConstants
from mlfcs.constraints.rotational import RotationalSumRuleResult
from mlfcs.physics.scph.solver import LoopSCPHResult, SCPHIteration
from mlfcs.physics.sscha.solver import SSCHAResult, SSCHAIteration
from mlfcs.physics.temperature import TemperatureSeriesResult
from mlfcs.structure.relation import StructureRelation, align_structures
```

这些对象适合结果分析、诊断和高级工作流；整数格、orbit、JAX packing 等内部实现不构成公共兼容承诺。

## 页面导航

- [结构与超胞](structures-api.md)
- [采样与 SSCHA](sscha-api.md)
- [有限差分](finite-difference-api.md)
- [力拟合](fitting-api.md)
- [力常数表示与 realization](force-constants-api.md)
- [ASE Calculator](calculator-api.md)
- [平移与旋转约束](constraints-api.md)
- [Loop SCPH](scph-api.md)
- [读取与写出](io-api.md)
- [单位、cutoff 与公共参数](units-and-parameters.md)
- [日志](logging.md)
- [异常与排错](exceptions.md)
- [版本与兼容范围](versioning.md)

## 通用约定

- `primitive` 描述无限晶体的 primitive site；`reference` 是本次计算的唯一显式超胞和原子顺序。
- 训练结构不会被静默重排，必须与 `reference` 的晶格、元素、原子数和顺序一致。
- MLFCS 内部 canonical IFC 和拟合坐标均为 Taylor 表示。
- 长度为 Å，力为 eV/Å，$n$ 阶 IFC 为 eV/Å$^n$，频率为 THz，温度为 K。
- 所有生成文件应显式指定格式；`.h5` 后缀本身不能区分 MLFCS、phonopy 和 phono3py schema。
