# MLFCS documentation

English | [中文](README_ZH.md)

MLFCS is documented in paired English and Chinese pages. The root
[README](../README.md) is the user entry point; the pages below cover complete workflows,
implementation details, validation, and compatibility decisions.

| Topic | English | 中文 |
|---|---|---|
| External VASP sow/reap workflow | [Guide](EXTERNAL_VASP_WORKFLOW.md) | [指南](EXTERNAL_VASP_WORKFLOW_ZH.md) |
| Technical architecture and algorithms | [Overview](TECHNICAL_OVERVIEW.md) | [概览](TECHNICAL_OVERVIEW_ZH.md) |
| Numerical validation and CI | [Validation](VALIDATION.md) | [数值验证](VALIDATION_ZH.md) |
| Translational and rotational sum rules | [Sum rules](SUM_RULES.md) | [求和规则](SUM_RULES_ZH.md) |
| Direct-calculator zero-step extrapolation | [Extrapolation](EXTRAPOLATION.md) | [零步长外推](EXTRAPOLATION_ZH.md) |
| Joint force-only FC2--FCn fitting | [Fitting](FITTING.md) | [仅力数据拟合](FITTING_ZH.md) |
| Fitting architecture versus ALAMODE | [Comparison](ALAMODE_COMPARISON.md) | [比较](ALAMODE_COMPARISON_ZH.md) |
| Native finite-temperature SSCHA | [SSCHA](SSCHA.md) | [SSCHA](SSCHA_ZH.md) |
| Previous and current implementations | [Comparison](OLD_NEW_COMPARISON.md) | [对比](OLD_NEW_COMPARISON_ZH.md) |
| Development roadmap | [Roadmap](ROADMAP.md) | [路线图](ROADMAP_ZH.md) |

Scientific fixtures and regeneration procedures are documented under
[`tests/reference`](../tests/reference/) and [`reference_tools`](../reference_tools/README.md).
Public APIs are demonstrated by executable files in [`examples`](../examples/).
The JOSS manuscript and its Chinese reading copy are indexed in [`paper`](../paper/README.md).
