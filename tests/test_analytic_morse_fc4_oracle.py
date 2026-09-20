from __future__ import annotations

from functools import cache
from itertools import product

import numpy as np
import sympy
from ase import Atoms
from ase.build import bulk
from ase.calculators.morse import MorsePotential
from ase.filters import FrechetCellFilter
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from supercell_helpers import make_supercell

from mlfcs import FiniteDifferenceCalculation

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


def calculation(displacement: float) -> FiniteDifferenceCalculation:
    primitive = analytic_primitive()
    return FiniteDifferenceCalculation(
        primitive,
        order=4,
        reference=make_supercell(primitive, SUPERCELL)[0],
        cutoff=CUTOFF,
        displacement=displacement,
    )


@cache
def _bond_fourth_derivative_evaluators():
    """Return one NumPy evaluator per fourth-derivative tensor component."""
    x, y, z, vx, vy, vz = sympy.symbols("x y z vx vy vz")
    distance = sympy.sqrt((vx + x) ** 2 + (vy + y) ** 2 + (vz + z) ** 2)
    exponential = sympy.exp(RHO0 * (1.0 - distance / R0))
    energy = EPSILON * exponential * (exponential - 2.0)
    origin = {x: 0, y: 0, z: 0}
    coordinates = (x, y, z)
    evaluators = []
    for indices in product(range(3), repeat=4):
        component = energy
        for index in indices:
            component = sympy.diff(component, coordinates[index])
        evaluators.append(sympy.lambdify((vx, vy, vz), component.subs(origin), "numpy"))
    return tuple(evaluators)


def _bond_fourth_derivative(vectors: np.ndarray) -> np.ndarray:
    """Exact fourth derivative of the Morse pair energy at zero displacement.

    The bond energy is differentiated symbolically in the three components of
    the relative displacement and evaluated at the origin, so this oracle is
    independent of any finite-difference or automatic-differentiation backend.
    """
    evaluators = _bond_fourth_derivative_evaluators()
    tensors = np.empty((len(vectors), 3, 3, 3, 3))
    for position, indices in enumerate(product(range(3), repeat=4)):
        tensors[(slice(None), *indices)] = evaluators[position](
            vectors[:, 0], vectors[:, 1], vectors[:, 2]
        )
    return tensors


def exact_sparse_fc4(
    calculation: FiniteDifferenceCalculation,
    clusters: np.ndarray,
) -> np.ndarray:
    """Evaluate exact FC4 on MLFCS clusters from an independent symbolic energy."""
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

    bond_fc4 = _bond_fourth_derivative(vectors)
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
    clusters = np.asarray(
        [
            [
                model.index.representative(int(sites[0])),
                *[
                    model.index.atom(int(site), translation)
                    for site, translation in zip(sites[1:], translations, strict=True)
                ],
            ]
            for sites, translations in zip(sparse.sites, sparse.translations, strict=True)
        ],
        dtype=np.int32,
    )
    return sparse.tensors, exact_sparse_fc4(model, clusters)


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual - expected
    return {
        "maximum": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(expected)),
        "correlation": float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1]),
    }
