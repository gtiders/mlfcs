# MLFCS 调用消费链

## 总览

三条主链（拟合、有限差分、periodic FC2 sidecar）共享 `InteractionSpace` 的枚举产物。
分歧仅在"观测→参数"机制：拟合走 design + streamed Gram + 求解器反演；有限差分走
差分模板直读 + ASR 投影；periodic sidecar 只挂靠 fitter，且它消费 `InteractionSpace`
的 `frame` 与 `primitive_orbit_space`，不消费 `orbit_space`。

## 流程图

```mermaid
flowchart TD
    subgraph INFRA["基础设施层"]
        PRIM["primitive + reference"]
        REL["StructureRelation / HNF 商格 / PeriodicIndex"]
        SYM["PrimitiveSymmetryOperations<br/>SymmetryOperations"]
        PRIM --> REL
        PRIM --> SYM
    end

    subgraph CORE["共享核心 InteractionSpace（每阶一个，两链共用）"]
        IS["InteractionSpace"]
        POS["primitive_orbit_space（懒加载）<br/>build_primitive_interaction_space<br/>集群枚举 + canonical key + |G|×n! images<br/>stabilizer Gram → invariant basis"]
        OS["orbit_space（懒加载）<br/>realize_orbit_space<br/>+ validate_realization_identifiability<br/>→ InteractionAliasingError"]
        IS --> POS
        IS --> OS
        POSS["generator_orbit 原型替代点<br/>（research/orbit_generators）"] -.-> POS
    end

    subgraph FIT["拟合链 ForceConstantFitter"]
        FT["ForceConstantFitter（每阶 InteractionSpace.from_frame）"]
        PAR["pack_order → order_tensors"]
        DES["OrderParameterization + ForceDesignOperator<br/>JAX design tile（消费 orbit.images）"]
        GRAM["streamed Gram + ASR/约束<br/>（translational.py 消费 images）"]
        SOL["求解器<br/>约束 LS / scaled group lasso"]
        LOW["backend.lower（Wick→Taylor lowering）"]
        EXP["expand_primitive_parameters<br/>（消费 basis/pivots → 稀疏 IFC）"]
        FC["ForceConstants.sparse（exact-R 可迁移）"]
        FT --> PAR
        FT --> IS
        POS --> DES
        OS --> DES
        DES --> GRAM
        GRAM --> SOL
        SOL --> LOW
        LOW --> EXP
        EXP --> FC
    end

    subgraph FD["有限差分链 FiniteDifferenceCalculation"]
        FDC["FiniteDifferenceCalculation（每阶 InteractionSpace）"]
        PLAN["orbit_space.displacement_keys<br/>× CentralDifferenceStencil 2^(n-1) 构型"]
        SOW["sow/reap：外部计算器力"]
        DIFF["差分导数直读填入 pivot<br/>+ project_acoustic_sum_rule（ASR 投影）"]
        FDC --> IS
        OS --> PLAN
        PLAN --> SOW
        SOW --> DIFF
        DIFF --> EXP
    end

    subgraph PER["periodic FC2 sidecar（仅 completion=True）"]
        PFC["SupercellHessianSpace.build(fc2_calculation)"]
        PB["_finite_pair_basis：pair orbit BFS<br/>+ stabilizer Gram → 有限对称基"]
        ASR["ASR 零空间 → compact_basis（H_SC^ASR）"]
        EX["_compact_exact_basis：exact 参数逐列 realize<br/>（消费 primitive_orbit_space）"]
        M["exact_map = B_SCᵀ B_E R_ASR"]
        SVD["SVD rank 检查 → InteractionAliasingError<br/>completion_basis = B_SC U[:, r:]"]
        APP["operator.append_periodic_fc2_block<br/>parameter_map block_diag"]
        PFC --> PB
        PB --> ASR
        ASR --> M
        PFC --> EX
        EX --> M
        M --> SVD
        SVD --> APP
        APP --> GRAM
        IS -- "frame + primitive_orbit_space（不消费 orbit_space）" --> PFC
    end

    subgraph OUT["结果消费"]
        WR["writers：phonopy / ALAMODE（拒绝 completion）<br/>ShengBTE / native HDF5 v3"]
        CALC["MLFCSCalculator"]
        PHYS["SSCHA / SCPH（q 点整数标签配对）"]
        FC --> WR
        FC --> CALC
        FC --> PHYS
        FC2C["ForceConstants.periodic_fc2_completion<br/>（source-bound sidecar）"]
        SVD --> FC2C
        FC2C --> CALC
        FC2C --> WR
    end
```

## 关键消费点

| 消费者 | 读的字段 | 位置 |
|---|---|---|
| ASR 约束 | `orbit.images` + `image.action` | `constraints/translational.py` |
| 参数化 / design | `orbits` / `dimension` / `images` | `fitting/parameterization.py`、`design_operator.py` |
| Gram 分块 | `orbit.dimension` | `fitting/gram_system.py` |
| Wick 跨阶 | `orbit.images` | `fitting/backends/wick/lowering.py` |
| IFC 展开 | `basis` / `pivots` | `force_constants/expansion.py` |
| periodic exact 映射 | `primitive_orbit_space` 逐列展开 | `force_constants/periodic_fc2.py` |
| FD 位移构型 | `pivots` → displacement_keys | `finite_difference/sampling.py` |
| FD 重构 | `pivots` 直读 + ASR | `finite_difference/reconstruction.py` |

## 要点

- 两条主链（拟合/差分）共享 `InteractionSpace` 的全部产物，分歧仅在观测→参数机制；
- periodic sidecar 只挂 fitter 的 Gram 阶段，且它自己又消费 `primitive_orbit_space`
  （exact 映射），即第三链建立在主链枚举产物之上，不消费 realize 后的 `orbit_space`；
- `generator_orbit` 原型（生成元 BFS + Schreier）的替代点只有一处：
  `build_primitive_interaction_space` 的 $|G| \times n!$ image 双重循环。
