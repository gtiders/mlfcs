# MLFCS

**MLFCS** 用于从晶体结构与原子受力构造高阶力常数模型。

从结构模型和用户明确给定的超胞出发，MLFCS 可以通过**有限差分**或**力拟合**构造统一的 `ForceConstants`，并进一步用于约束处理、有限温度重整和晶格动力学计算。

!!! tip "两条主要工作流"

    **有限差分**：结构 → 位移构型 → 力 → `ForceConstants`

    **力拟合**：结构 + 采样构型 + 力 → `ForceConstants`

## 从这里开始

如果是第一次使用 MLFCS，推荐按照下面的顺序阅读：

1. [核心概念](core-concepts.md)
   了解 `ClusterSpace`、`Supercell`、`ClusterMap` 以及结构可辨识性。
2. 选择力常数构造方式：
   - [有限差分](finite-difference.md)：从位移构型和原子力计算力常数。
   - [力拟合](fitting.md)：从一组结构与原子力拟合力常数。
3. [力常数](force-constants.md)
   查看、约束、变换和导出 `ForceConstants` 模型。

## 进阶功能

### 有限温度

- [SCPH](scph.md)：自洽声子计算与有限温度重整。
- [SSCHA](sscha.md)：基于随机采样的有限温度有效谐波模型。

### 极性材料

- [长程力](long-range-forces.md)：处理偶极长程相互作用及长程/短程力常数的分离与组合。

### 常见问题

- [Q&A](Q&A.md)：力的存储方式、原子顺序、结构可辨识性、求解器以及常见参数选择。

## 核心对象

MLFCS 的基本对象关系可以概括为：

```text
原胞 ASE Atoms ──→ ClusterSpace ──┐
                                  ├──→ ClusterMap → 有限差分/拟合 → ForceConstants
超胞 ASE Atoms ──→ Supercell ────┘
```

其中：

- `Atoms` 描述晶体结构；
- `ClusterSpace` 定义需要考虑的力常数空间；
- `Supercell` 定义实际进行位移或力计算的几何结构；
- `ClusterMap` 建立对称性约化后的参数与实际超胞之间的映射；
- `ForceConstants` 保存最终得到的力常数模型。

## 单位与结构约定

MLFCS 统一使用 ASE 的 `Atoms` 对象作为结构表示。

| 物理量 | 单位 |
| --- | --- |
| 长度 | Å |
| 能量 | eV |
| 力 | eV/Å |
| $n$ 阶力常数 | eV/Å$^n$ |

!!! important "超胞与原子顺序"

    用户提供的超胞结构以及其中的**原子顺序**，定义了力计算、参数映射和力常数重建所使用的几何约定。在整个工作流中应保持这一顺序一致。
