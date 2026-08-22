# 变更记录

[English](CHANGELOG.md)

本文件记录面向用户的重要变化，版本遵循语义化版本约定。

## 未发布

### 新增

- 基于 compact FC2 的原生相容 q 点采样，同时支持量子和经典谐振系综；
- 显式虚频策略、频率过滤、采样诊断和可选的逐原子径向位移裁剪，默认不裁剪；
- 解析谐振模型测试以及仅开发环境运行的独立 phonopy 采样参考。
- 使用 phonopy 官方 pypolymlp 势函数和夹具的端到端 KCl SSCHA 参考。

### 变化

- `mlfcs.sscha` 改用共享的 MLFCS 对称约化 Gram 拟合器求解 FC2，并复用统一
  力常数 I/O；phonopy 和 symfc 不再是运行时依赖。
- canonical 迭代现在派生相互独立且可复现的子种子；笛卡尔初始化轮不再报告统计上
  未定义的 SSCHA 自由能。

## 3.1.0 — 2026-08-09

### 新增

- 二阶力常数可通过 `rotational_sum_rule=True` 主动开启 Born–Huang 旋转求和规则；在
  联合阶数 API 能表达高阶耦合约束前，其他阶数会明确拒绝该选项；
- 平移与旋转约束使用一次联合稀疏 LSMR 投影；
- 求和规则投影前后默认报告 phonopy 风格的最大 drift；
- `cutoff=None` 表示当前超胞可枚举的最大相互作用半径；
- 新增中英文配对的求和规则文档。

### 变化

- 所有参数规模的平移 ASR 统一使用稀疏、矩阵无关的 LSMR 路径，删除稠密 Gram
  构造和规模切换阈值。

## 3.0.0 — 2026-08-03

3.0 是 MLFCS 的完整 ASE-first 重构：原有的分阶实现被统一的阶数参数化 API 和数值
流程取代。

### 新增

- `order >= 2` 的统一有限差分力常数流程；
- ASE Calculator 直接运行和确定性的外部 `sow()` / `reap()`；
- 递归中心有限差分模板和位移键去重；
- 对称性展开的稀疏力常数与惰性稠密物化；
- Gram 零空间和稀疏 LSMR 实现的严格平移 ASR；
- JAX 高阶张量操作的 CPU/GPU 选择；
- 任意阶通用稀疏 HDF5；
- phonopy FC2、phono3py FC3 HDF5 以及 ShengBTE FC3/FC4 输出；
- 默认保真的 ShengBTE 周期几何和显式 thirdorder 兼容模式；
- 可选的 phonopy/symfc 随机有效谐波模块；
- phonopy、phono3py、hiphive 转换、ShengBTE 与解析 Morse FC4 科学参考；
- Python 3.12/3.13 串行科学 CI。

### 变化

- 公共接口统一使用 ASE `Atoms` 和用户持有的 ASE calculator；
- 力生成不再绑定某个电子结构程序或机器学习势；
- 重建和默认保真导出共享同一个周期团簇几何；
- 3.0 只提供 Python API，不再提供 CLI。

### 兼容性

- thirdorder 的 sow 顺序和 ShengBTE 布局必须显式请求；
- 旧脚本需要迁移到 `ForceConstantCalculation`、`sow()`、`reap()` 或 `run()`；
- `v3.0.0` 之前的标签属于旧实现或开发快照，仅为追溯保留，不属于 3.0 API 契约。
