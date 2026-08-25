---
title: 读取与写出 API
audience:
  - user
  - developer
status: stable
code_verified: 4.0.0a6
---

# 读取与写出 API

## `read_hdf5`

```python
read_hdf5(source: str | Path) -> ForceConstants
```

读取 MLFCS 原生 HDF5 v3。文件保存 primitive、source/reference 关系、exact sites/translations/tensors 和
metadata。它不是 phonopy/phono3py 的稠密 HDF5。路径不存在、schema 不是 v3、shape 损坏或结构映射不一致
时抛出异常。

启用 periodic FC2 completion 时，同一 v3 文件会额外保存 source reference、compact periodic Hessian 与
rank report。该扩展是可选的；普通 exact-$R$ 文件的结构不变。读取后 `sparse[2]` 仍只包含 transferable
exact-$R$ IFC，`materialize(2)` 才组合 source-bound completion。

## `write_force_constants`

```python
write_force_constants(
    force_constants: ForceConstants,
    target: str | Path,
    *,
    format: str,
    order: int | None = None,
    primitive: Atoms | None = None,
    supercell: Atoms | None = None,
) -> None
```

| 参数 | 含义 |
|---|---|
| `force_constants` | 要写出的 Taylor IFC。 |
| `target` | 输出文件路径；父目录必须由调用者准备。 |
| `format` | 必须显式给出，见下表。 |
| `order` | 单阶格式的 IFC 阶；某些格式固定阶数。 |
| `primitive` | 格式或 target realization 所需的显式 primitive。 |
| `supercell` | 格式所需的显式目标超胞；不会由矩阵隐式构造。 |

| `format` | 支持阶 | 表示与要求 |
|---|---|---|
| `hdf5` | 文件中全部阶 | MLFCS v3 sparse；`order` 应省略 |
| `phonopy` | 仅 FC2 | 文本 `FORCE_CONSTANTS`，按当前或给定 target 稠密化 |
| `phonopy_hdf5` | 仅 FC2 | phonopy 稠密 HDF5，需要有效超胞映射 |
| `phono3py_hdf5` | 仅 FC3 | phono3py 稠密 HDF5，内存开销可能很大 |
| `shengbte` | FC3 或 FC4 | 直接消费 lattice-labelled sparse IFC，避免全张量稠密化 |
| `alamode` | 一个或多个已有阶 | XML，执行格式专属 27-image 编码验证 |

格式名未知、order 缺失/不存在或格式不支持所选阶时抛出 `ValueError`。ShengBTE 要求 sparse exact-$R$ IFC；
ALAMODE 若无法在目标胞内表达所需镜像，会抛出 `AlamodeMirrorImageError`，而不是写出近似文件。

带 periodic FC2 completion 的结果可写为 native HDF5、phonopy 文本或 phonopy HDF5；后二者写出 total
source Hessian。ShengBTE 的高阶 exact-$R$ 输出不受影响。ALAMODE FC2 无法表达这一 source-bound sidecar，
因此会明确拒绝，而不是静默丢弃。

```python
write_force_constants(fc, "mlfcs.h5", format="hdf5")
write_force_constants(fc, "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
write_force_constants(fc, "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
```

文件扩展名不参与格式推断。writer 会缓存同一 target 的 export view，因此连续写多个格式无需重复构造周期映射。
