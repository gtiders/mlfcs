# 原生 HDF5 v2

原生 HDF5 v2 是唯一的 MLFCS 交换 schema。它显式保存稀疏支撑，不能通过张量是否为零来猜测条目是否存在；
只有要求稠密数组的目标 writer 才会显式 materialize。

pre-v2 文件会明确报告 schema 不支持；没有会猜测旧原子语义的迁移 reader。
