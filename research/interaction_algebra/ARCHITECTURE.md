# Interaction algebra 生产架构建议

## 返回与消费边界

`PrimitiveInteractionSpace` 表示无限 primitive lattice 上可迁移的 exact-$R$ 参数空间；
`RealizedInteractionSpace` 表示它在一个 reference supercell 中的 realization。后者只消费前者，
不重新定义 orbit 参数。

```python
def build_primitive_interaction_space(
    primitive: Atoms,
    *,
    order: int,
    cutoff: float | None,
    max_body_order: int | None,
    symprec: float,
    tolerance: float = 1e-9,
    symmetry: PrimitiveSymmetryOperations | None = None,
) -> PrimitiveInteractionSpace: ...
```

```python
def realize_interaction_space(
    space: PrimitiveInteractionSpace,
    index: PeriodicIndex,
    *,
    validate_identifiability: bool = True,
    tolerance: float = 1e-10,
) -> RealizedInteractionSpace: ...
```

```python
class InteractionSpace:
    @property
    def primitive_orbit_space(self) -> PrimitiveInteractionSpace: ...

    @property
    def realized_orbit_space(self) -> RealizedInteractionSpace: ...
```

空间对象从顶层 `mlfcs` 导出。orbit/image 类型从 `mlfcs.interactions` 导出，并通过空间对象的
`orbits`/`images` 属性访问，不加入顶层命名空间。

## 领域数据类型

```python
@dataclass(frozen=True, slots=True)
class PrimitiveInteractionOrbit:
    representative: InteractionKey
    basis: NDArray[np.float64]
    pivots: NDArray[np.int32]
    images: tuple[PrimitiveOrbitImage, ...]
```

```python
@dataclass(frozen=True, slots=True)
class PrimitiveOrbitImage:
    key: InteractionKey
    action: TensorAction
```

```python
@dataclass(frozen=True, slots=True)
class RealizedInteractionOrbit:
    representative: tuple[int, ...]
    basis: NDArray[np.float64]
    pivots: NDArray[np.int32]
    images: tuple[RealizedOrbitImage, ...]
```

```python
@dataclass(frozen=True, slots=True)
class RealizedOrbitImage:
    cluster: tuple[int, ...]
    action: TensorAction
```

image 必须保存 canonical-to-image `TensorAction`，而不是只保存 transported basis。不同作用路径
可以相差 stabilizer，因此 action 等价性应在 invariant basis 上判定。

## 内部代数接口

研究证明生产接口还需要 codec 明确状态编码和 canonical ordering，不能只提供抽象 `transform()`：

```python
class StateCodec(Protocol[StateT]):
    width: int
    canonical_columns: tuple[int, ...]

    def encode(self, state: StateT) -> NDArray[np.int64]: ...
    def decode(self, row: NDArray[np.int64]) -> StateT: ...
```

```python
class GeneratorAction(Protocol):
    name: str
    action: TensorAction

    def transform(self, states: NDArray[np.int64]) -> NDArray[np.int64]: ...
```

```python
@dataclass(frozen=True, slots=True)
class IndexedOrbitResult(Generic[StateT]):
    canonical: StateT
    states: tuple[StateT, ...]
    actions: tuple[TensorAction, ...]
    constraint_gram: NDArray[np.float64]
```

```python
def traverse_indexed_orbit(
    seed: StateT,
    generators: tuple[GeneratorAction, ...],
    *,
    codec: StateCodec[StateT],
    order: int,
    seed_basis: NDArray[np.float64],
    tolerance: float = 1e-9,
) -> IndexedOrbitResult[StateT]: ...
```

## 目录建议

```text
src/mlfcs/interactions/
├── __init__.py
├── models.py
├── space.py
├── keys.py
├── realization.py
├── algebra/
│   ├── actions.py
│   ├── generators.py
│   ├── indexed_orbit.py
│   └── invariants.py
└── primitive/
    ├── candidates.py
    └── builder.py
```

periodic harmonic 的稳定物理对象是 source-bound compact Hessian response，不是 interaction
orbit。建议最终放在 `force_constants/finite_harmonic/`，其中 builder 可消费
`interactions.algebra`，但 response 不反向依赖 `InteractionSpace`，也不写入 exact-$R$ sparse IFC。

## 诊断

群阶、生成元数、edge 数、rank、completion dimension 和 tolerance 统一写入 `mlfcs` logger。
不提供诊断函数，不把临时 rank report 或 SVD basis 写入物理对象。若正式迁移，再一次性删除
`PeriodicFC2RankReport` 及 HDF5 `rank_diagnostics`；本研究阶段不修改 schema。

