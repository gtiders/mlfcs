# MLFCS 文档

[English](README.md) | 中文

MLFCS 的用户文档均提供英文和中文版本。仓库根目录的
[README](../README_ZH.md) 是用户入口；下列页面分别说明完整工作流、实现细节、
数值验证和兼容性决策。

| 主题 | 中文 | English |
|---|---|---|
| 外部 VASP sow/reap 工作流 | [指南](EXTERNAL_VASP_WORKFLOW_ZH.md) | [Guide](EXTERNAL_VASP_WORKFLOW.md) |
| 技术架构与算法 | [概览](TECHNICAL_OVERVIEW_ZH.md) | [Overview](TECHNICAL_OVERVIEW.md) |
| 数值验证与 CI | [数值验证](VALIDATION_ZH.md) | [Validation](VALIDATION.md) |
| 平移与旋转求和规则 | [求和规则](SUM_RULES_ZH.md) | [Sum rules](SUM_RULES.md) |
| 直接 calculator 零步长外推 | [零步长外推](EXTRAPOLATION_ZH.md) | [Extrapolation](EXTRAPOLATION.md) |
| FC2--FCn 联合仅力数据拟合 | [仅力数据拟合](FITTING_ZH.md) | [Fitting](FITTING.md) |
| 与 ALAMODE 的拟合架构比较 | [比较](ALAMODE_COMPARISON_ZH.md) | [Comparison](ALAMODE_COMPARISON.md) |
| 可选有限温度 SSCHA | [SSCHA](SSCHA_ZH.md) | [SSCHA](SSCHA.md) |
| 新旧实现差异 | [对比](OLD_NEW_COMPARISON_ZH.md) | [Comparison](OLD_NEW_COMPARISON.md) |

科学基准夹具及其再生成方法位于 [`tests/reference`](../tests/reference/) 和
[`reference_tools`](../reference_tools/README_ZH.md)，公共 API 的可执行示例位于
[`examples`](../examples/)。
JOSS 英文稿及中文阅读版见 [`paper`](../paper/README_ZH.md)。
