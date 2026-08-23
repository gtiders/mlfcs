---
title: 单位与参数
audience:
  - user
status: stable
code_verified: 4.0.0a4
---

# 单位与参数

| 量 | 单位或约定 |
|---|---|
| 晶格、坐标、位移 | Å |
| 力 | eV/Å |
| n 阶 IFC | eV/Åⁿ |
| 正 cutoff | Å 半径 |
| 负 cutoff | 邻居壳层编号 |
| `None` cutoff | 当前 reference supercell 中 exact-$R$ 交互的最大无周期像歧义半径；从第一个歧义边界减去 $0.01$ Å |
| SCPH 容差 | 频率变化 RMS，单位 THz |

JAX kernel 使用 64 位浮点。`mixing` 是数值松弛系数，不是物理参数；`tolerance` 是停止判据，不会把
力常数归零，也不会改变拟合支撑域。

`cutoff=None` 不表示无穷大作用范围，也不等价于 ALAMODE/phonopy 对当前有限超胞的完整周期化 FC2。它为 primitive exact-$R$ 模型选择当前 source reference 不会把同一原子对的不同周期像同时收入的最大安全半径。对 KCl 等存在偶极长程尾部的极性晶体，`None` 不能代替 source-supercell 收敛检查或解析长程静电分解。
