from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.morse import MorsePotential
from ase.filters import FrechetCellFilter
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from supercell_helpers import make_supercell

from mlfcs import ForceConstantCalculation

EPSILON = 1.0
RHO0 = 6.0
R0 = 1.0
RCUT1 = 1.15
RCUT2 = 1.30
SUPERCELL = (3, 3, 3)
CUTOFF = 1.1


def calculator() -> MorsePotential:
    return MorsePotential(
        epsilon=EPSILON,
        rho0=RHO0,
        r0=R0,
        rcut1=RCUT1,
        rcut2=RCUT2,
    )


def analytic_primitive() -> Atoms:
    """Return the exact nearest-neighbour equilibrium FCC primitive cell."""
    return bulk("Ar", "fcc", a=np.sqrt(2.0) * R0)


def relaxed_primitive() -> tuple[Atoms, int]:
    """Relax an intentionally strained cell using only the ASE Morse calculator."""
    atoms = bulk("Ar", "fcc", a=1.5 * R0)
    atoms.calc = calculator()
    optimizer = BFGS(
        FrechetCellFilter(atoms, hydrostatic_strain=True),
        logfile=None,
    )
    optimizer.run(fmax=1.0e-12)
    return atoms, optimizer.get_number_of_steps()


def calculation(displacement: float) -> ForceConstantCalculation:
    primitive = analytic_primitive()
    return ForceConstantCalculation(
        primitive,
        order=4,
        reference=make_supercell(primitive, SUPERCELL)[0],
        cutoff=CUTOFF,
        displacement=displacement,
        report_cutoff=False,
        verbose=False,
    )


@partial(jax.jit, static_argnums=())
def _bond_fourth_derivative(vectors: jax.Array) -> jax.Array:
    def energy(relative_displacement: jax.Array, vector: jax.Array) -> jax.Array:
        distance = jnp.linalg.norm(vector + relative_displacement)
        exponential = jnp.exp(RHO0 * (1.0 - distance / R0))
        return EPSILON * exponential * (exponential - 2.0)

    derivative = energy
    for _ in range(4):
        derivative = jax.jacfwd(derivative, argnums=0)
    return jax.vmap(lambda vector: derivative(jnp.zeros(3), vector))(vectors)


def exact_sparse_fc4(
    calculation: ForceConstantCalculation,
    clusters: np.ndarray,
) -> np.ndarray:
    """Evaluate exact FC4 on MLFCS clusters from an independent JAX energy."""
    supercell = calculation.supercell
    first, second, shifts = neighbor_list("ijS", supercell, RCUT2 * R0)
    unique = first < second
    first = first[unique]
    second = second[unique]
    shifts = shifts[unique]
    vectors = (
        supercell.positions[second] + shifts @ supercell.cell.array - supercell.positions[first]
    )
    # The chosen cutoff window excludes second neighbours and is identically one
    # for every nearest-neighbour bond in the finite-difference neighbourhood.
    assert len(vectors) == 162
    assert np.allclose(np.linalg.norm(vectors, axis=1), R0, atol=1.0e-12, rtol=0)

    bond_fc4 = np.asarray(_bond_fourth_derivative(jnp.asarray(vectors)))
    tensors = np.zeros((len(clusters), 3, 3, 3, 3))
    for cluster_index, cluster in enumerate(clusters):
        for atom_a, atom_b, derivative in zip(first, second, bond_fc4, strict=True):
            signs = []
            for atom in cluster:
                if atom == atom_a:
                    signs.append(-1.0)
                elif atom == atom_b:
                    signs.append(1.0)
                else:
                    break
            else:
                tensors[cluster_index] += np.prod(signs) * derivative
    return tensors


def finite_difference_fc4(displacement: float) -> tuple[np.ndarray, np.ndarray]:
    model = calculation(displacement)
    result = model.run(calculator(), acoustic_sum_rule=False)
    sparse = result.sparse[4]
    return sparse.tensors, exact_sparse_fc4(model, sparse.clusters)


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual - expected
    return {
        "maximum": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(expected)),
        "correlation": float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1]),
    }
