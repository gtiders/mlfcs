# Loop-SCPH

`LoopSCPH` 对 FC2 应用静态四阶 loop 修正，返回与温度相关的有效 FC2。FC2 和 FC4 是两个独立的
`ForceConstants` 对象，但必须描述同一个 primitive/reference frame。

```python
result = LoopSCPH(
    fc2=fc2, fc4=fc4, temperature=600,
    interpolation_mesh=(3, 3, 3), scph_mesh=(6, 6, 6),
    mixing=0.1, tolerance=1e-10, max_iterations=100,
).run()
```

`mixing` 是数值阻尼。当插值网格上的频率变化 RMS 小于 `tolerance` 且所有频率平方非负时，
迭代才判定为收敛。当前实现只包含 loop，不包含频率相关的 bubble 自能。
