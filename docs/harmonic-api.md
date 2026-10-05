# Harmonic frequencies and atomic masses

## Inspect or replace masses

`ClusterSpace` copies per-site masses from its input ASE `Atoms`, in atomic mass units. `cs.masses` is a readonly array of shape `(cs.n_atoms,)`. Changing the input atoms or the detached `cs.primitive_atoms` snapshot does not change the space.

```python
from mlfcs import ForceConstants
from mlfcs.phonon import Harmonic

heavy_cs = model.cluster_space.with_masses(2 * model.cluster_space.masses)
heavy_model = ForceConstants(heavy_cs, model.coefficients)
heavy_harmonic = Harmonic(heavy_model)
```

`with_masses(masses)` requires finite, strictly positive values of shape `(n_atoms,)`. It copies only the mass assignment and shares the existing readonly geometry, symmetry, orbits and parameterization. It does not repeat spglib preprocessing, cluster enumeration or exact algebra. The original space and model remain unchanged.

Masses do not change the physical parameter layout. Physical energy derivatives can therefore be rebound to the new space without refitting. Native HDF5 force-constant files retain the mass assignment. `Harmonic.masses` exposes a readonly snapshot without a setter: modifying masses means constructing a new space and harmonic model, not mutating cached mass weights.

## Frequencies at explicit q points

```python
harmonic = Harmonic(model)  # model must contain FC2
one = harmonic.frequencies([0.25, 0.0, 0.0])
many = harmonic.frequencies([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
matrices = harmonic.dynamical_matrices([[0.25, 0.0, 0.0]])
```

q coordinates are fractional reciprocal coordinates in the model's primitive basis. A single `(3,)` input returns frequencies of shape `(3*n_atoms,)`; a nonempty `(nq, 3)` batch returns `(nq, 3*n_atoms)`. Modes are ordered by ascending dynamical-matrix eigenvalue, independently at each point; no band connectivity is inferred.

Frequencies are ordinary frequencies in **THz**, including the $1/(2\pi)$ conversion from angular frequency. Negative values represent imaginary modes as $-\sqrt{|\lambda|}$ times the conversion factor; they are not clipped to zero. Multiplying all masses by $a>0$ divides all frequencies by $\sqrt a$ while leaving FC coefficients unchanged.

`dynamical_matrices(qpoints)` returns complex Hermitian matrices in positional Fourier gauge, with shape `(3*n_atoms, 3*n_atoms)` for one point or `(nq, 3*n_atoms, 3*n_atoms)` for a batch. Entries have units eV/(angstrom²·atomic mass unit), before frequency conversion. The former name `matrices()` is removed.

## Direct mesh calculation

```python
result = harmonic.mesh((8, 8, 8), time_reversal=True)
result.qpoints            # fractional irreducible representatives
result.weights            # integer member counts, not normalized probabilities
result.frequencies_thz    # signed THz at representatives
result.mesh_matrix        # exact integer row-cell matrix
full = result.full_frequencies()
```

`mesh` accepts positive integer sizes `(nx, ny, nz)`, a nonsingular integer 3×3 supercell matrix, or an existing `QGrid`. The matrix follows `supercell_cell = matrix @ primitive_cell`; a size triple becomes a diagonal matrix. This is a Gamma-centered exact mesh. Shifted meshes are not provided.

`HarmonicMeshResult` keeps readonly arrays. Its weights sum to the full mesh size, and frequencies have shape `(n_representatives, 3*n_atoms)`. Only representative Fourier matrices are evaluated. `full_frequencies()` allocates a new frequency array in the lexicographic exact-label order used by `QGrid`; its corresponding coordinates are `QGrid(result.mesh_matrix).points`.

Explicit points permit different masses on structurally equivalent sites. Mesh reduction uses only structural operations that preserve the actual per-site masses, plus time reversal when requested. It does not change the cluster-space symmetry or FC basis. The same rule applies to SCPH.

Advanced callers may still pass `QStars` to `frequencies()` or `dynamical_matrices()`. Callers must ensure its operations belong to the model's primitive symmetry and preserve masses. External star/model compatibility is not checked; Harmonic.mesh() constructs the appropriate subgroup automatically.

All calculations continue to use three-dimensional, fully periodic cells. This API change does not introduce partial-PBC or vacuum validation.
