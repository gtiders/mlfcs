# 长程静电修正

本主题使用同源的 Born 有效电荷、电子介电张量、结构和参考力，展示极性晶体中解析长程响应与短程力常数拟合的分离。

- [`NaCl/`](NaCl/)：复现 hiPhive 官方 NaCl 数据，并对比 phonopy、hiPhive 与 MLFCS。

当前案例通过 phonopy 构造 Gonze–Lee 长程 FC2；它是教学和数值交叉验证，不表示 MLFCS 已经提供原生长程静电计算内核。
