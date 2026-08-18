# 单位与参数

| 量 | 单位或约定 |
|---|---|
| 晶格、坐标、位移 | Å |
| 力 | eV/Å |
| n 阶 IFC | eV/Åⁿ |
| 正 cutoff | Å 半径 |
| 负 cutoff | 邻居壳层编号 |
| `None` cutoff | 参考超胞可枚举的最大半径 |
| SCPH 容差 | 频率变化 RMS，单位 THz |

JAX kernel 使用 64 位浮点。`mixing` 是数值松弛系数，不是物理参数；`tolerance` 是停止判据，不会把
力常数归零，也不会改变拟合支撑域。
