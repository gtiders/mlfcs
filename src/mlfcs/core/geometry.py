from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from ase import Atoms


@dataclass(frozen=True, slots=True)
class SupercellIndex:
    primitive: np.ndarray
    translations: np.ndarray
    repeats: tuple[int, int, int]

    @property
    def n_primitive(self) -> int:
        return int(self.primitive.max()) + 1

    def atom(self, primitive: int, translation: np.ndarray) -> int:
        wrapped = np.mod(translation, self.repeats)
        cell_id = int(wrapped[0] + self.repeats[0] * (wrapped[1] + self.repeats[1] * wrapped[2]))
        return cell_id * self.n_primitive + int(primitive)

    def translate_atom(self, atom: int, shift: np.ndarray) -> int:
        return self.atom(int(self.primitive[atom]), self.translations[atom] + shift)

    def anchor(self, cluster: tuple[int, ...]) -> tuple[int, ...]:
        shift = -self.translations[cluster[0]]
        return tuple(self.translate_atom(atom, shift) for atom in cluster)

    @property
    def grouped_permutation(self) -> np.ndarray:
        """Map primitive-atom-grouped order to internal cell-major order."""
        n_cells = int(np.prod(self.repeats))
        return np.arange(n_cells * self.n_primitive).reshape(n_cells, self.n_primitive).T.ravel()

    @property
    def internal_from_grouped(self) -> np.ndarray:
        """Indices selecting grouped data to recover internal order."""
        return np.argsort(self.grouped_permutation)

    def group_atoms(self, supercell: Atoms) -> Atoms:
        if len(supercell) != len(self.primitive):
            raise ValueError("supercell length does not match index")
        return supercell[self.grouped_permutation]


def make_supercell(atoms: Atoms, repeats: tuple[int, int, int]) -> tuple[Atoms, SupercellIndex]:
    """Build a diagonal supercell with deterministic cell-major atom ordering."""
    if not all(atoms.pbc):
        raise ValueError("force constants require periodic boundary conditions")
    primitive = atoms.copy()
    primitive.wrap()
    primitive_scaled = primitive.get_scaled_positions()
    translations = np.asarray(
        list(product(range(repeats[2]), range(repeats[1]), range(repeats[0])))
    )
    translations = translations[:, ::-1]  # x is the fastest cell index
    positions = np.concatenate(
        [primitive.positions + shift @ primitive.cell for shift in translations]
    )
    numbers = np.tile(primitive.numbers, len(translations))
    matrix = np.diag(repeats)
    supercell = Atoms(numbers=numbers, positions=positions, cell=matrix @ primitive.cell, pbc=True)
    atom_translations = np.repeat(translations, len(primitive), axis=0)
    primitive_indices = np.tile(np.arange(len(primitive), dtype=np.int32), len(translations))
    supercell.arrays["primitive_index"] = primitive_indices
    supercell.arrays["cell_translation"] = atom_translations.astype(np.int32)
    supercell.arrays["primitive_scaled_position"] = np.tile(
        primitive_scaled, (len(translations), 1)
    )
    return supercell, SupercellIndex(primitive_indices, atom_translations, repeats)


def neighbor_shell_cutoff(
    supercell: Atoms,
    index: SupercellIndex,
    shell: int,
    *,
    report: bool = True,
) -> float:
    """Return the midpoint after a one-based neighbor shell, in angstrom."""
    if shell < 1:
        raise ValueError("neighbor shell must be positive")
    distances = supercell.get_all_distances(mic=True)
    maximum_shell, maximum_radius = neighbor_shell_limit(supercell, index, distances=distances)
    if report:
        print(
            "Supercell neighbor limit: "
            f"maximum shell = {maximum_shell}, "
            f"maximum cutoff radius = {maximum_radius:.10f} Å"
        )
    if shell > maximum_shell:
        raise ValueError(
            f"neighbor shell {shell} exceeds this supercell's enumerable maximum "
            f"of {maximum_shell} (cutoff radius {maximum_radius:.10f} Å)"
        )
    candidates: list[float] = []
    for atom in range(index.n_primitive):
        unique = _unique_distances(distances[atom])
        if len(unique) <= shell:
            candidates.append(unique[-1] * 1.1)
        else:
            candidates.append((unique[shell - 1] + unique[shell]) / 2.0)
    selected_radius = float(max(candidates))
    if report:
        print(
            f"Selected neighbor cutoff: shell = {shell}, cutoff radius = {selected_radius:.10f} Å"
        )
    return selected_radius


def neighbor_shell_limit(
    supercell: Atoms,
    index: SupercellIndex,
    *,
    distances: np.ndarray | None = None,
) -> tuple[int, float]:
    """Return the largest shell and cutoff enumerable in the MIC supercell."""
    if distances is None:
        distances = supercell.get_all_distances(mic=True)
    shells = [_unique_distances(distances[atom]) for atom in range(index.n_primitive)]
    maximum_shell = min(len(values) for values in shells)
    if maximum_shell < 1:
        raise ValueError("supercell is too small to contain a reliable neighbor shell")
    boundaries = []
    for values in shells:
        if len(values) <= maximum_shell:
            boundaries.append(values[-1] * 1.1)
        else:
            boundaries.append((values[maximum_shell - 1] + values[maximum_shell]) / 2.0)
    maximum_radius = float(max(boundaries))
    return maximum_shell, maximum_radius


def resolve_cutoff(
    supercell: Atoms,
    index: SupercellIndex,
    cutoff: float | None,
    *,
    report: bool = True,
) -> float:
    if cutoff is None:
        maximum_shell, maximum_radius = neighbor_shell_limit(supercell, index)
        if report:
            print(
                "Supercell neighbor limit: "
                f"maximum shell = {maximum_shell}, "
                f"maximum cutoff radius = {maximum_radius:.10f} Å"
            )
            print(f"Selected maximum cutoff radius: {maximum_radius:.10f} Å")
        return maximum_radius
    value = float(cutoff)
    if value < 0 and value.is_integer():
        return neighbor_shell_cutoff(supercell, index, -int(value), report=report)
    if value <= 0:
        raise ValueError("cutoff must be a positive distance or negative integer shell")
    return value


def _unique_distances(
    values: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> list[float]:
    """Group numerically split but physically equivalent neighbor shells."""
    result: list[float] = []
    for value in np.sort(values):
        if value <= atol:
            continue
        if not result or not np.isclose(value, result[-1], atol=atol, rtol=rtol):
            result.append(float(value))
    if not result:
        raise ValueError("no periodic neighbors found; supercell is too small")
    return result
