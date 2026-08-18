# MLFCS

MLFCS 是一个以 ASE 为边界的 Python 力常数库，用于从原子力计算对称约化的力常数。
它支持有限差分和仅力数据拟合，从 FC2 到所有实现的更高阶，并使用稀疏原生存储和显式导出视图。

## 按目标选择工作流

| 目标 | 从这里开始 |
|---|---|
| 使用 ASE calculator 计算 FC2/FC3/FC4 | [有限差分](workflows/finite-difference.md) |
| 使用 VASP 或其他外部力程序 | [外部计算器](workflows/external-calculators.md) |
| 从位移或 MD 快照拟合力常数 | [仅力数据拟合](workflows/fitting.md) |
| 通过采样生成温度相关 FC2 | [SSCHA](workflows/sscha.md) |
| 应用四阶 loop 修正 | [Loop-SCPH](workflows/scph.md) |
| 导出到声子或输运软件 | [格式](formats/index.md) |

## 开始前的三条规则

1. 先决定结果由哪个后处理软件读取。
2. 尽可能使用该软件提供的 primitive 和参考超胞。
3. 从结构生成到力收集始终保持参考原子顺序。

MLFCS 会在导出时验证严格等价的表示，但不会静默改变原胞定义、放大超胞或整体旋转结构。

## 安装

```bash
uv sync
```

基础包不会安装 phonopy、phono3py、ShengBTE、ALAMODE 或具体 calculator；只有实际使用相应工作流时才安装它们。

最小完整示例见[入门](getting-started/index.md)，可复现材料流程见[案例](cases/index.md)。
