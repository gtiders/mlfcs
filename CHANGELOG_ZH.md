# 变更记录

[English](CHANGELOG.md)

本文件记录面向用户的重要变化，版本遵循语义化版本约定。

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
