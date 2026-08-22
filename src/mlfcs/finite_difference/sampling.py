from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from ase import Atoms

from mlfcs.core.orbits import OrbitSpace
from mlfcs.finite_difference.stencil import CentralDifferenceStencil

DisplacementKey = tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class DisplacementConfiguration:
    key: DisplacementKey
    signs: np.ndarray
    displacement: np.ndarray


@dataclass(frozen=True, slots=True)
class DisplacementPlan:
    supercell: Atoms
    configurations: tuple[DisplacementConfiguration, ...]
    stencil: CentralDifferenceStencil

    def __len__(self) -> int:
        return len(self.configurations)

    @property
    def hash(self) -> str:
        digest = sha256()
        digest.update(str(self.stencil.derivative_order).encode())
        digest.update(np.float64(self.stencil.step).tobytes())
        for configuration in self.configurations:
            digest.update(repr(configuration.key).encode())
            digest.update(np.ascontiguousarray(configuration.displacement).tobytes())
        return digest.hexdigest()

    def atoms(self, index: int) -> Atoms:
        atoms = self.supercell.copy()
        atoms.positions += self.configurations[index].displacement
        atoms.info["mlfcs_configuration_id"] = index
        atoms.info["mlfcs_plan_hash"] = self.hash
        atoms.info["mlfcs_atom_order"] = "internal"
        atoms.arrays["mlfcs_displacement"] = self.configurations[index].displacement.copy()
        return atoms

    def __iter__(self):
        for index in range(len(self)):
            yield self.atoms(index)

    def contract_forces(self, forces: np.ndarray) -> dict[DisplacementKey, np.ndarray]:
        values = np.asarray(forces, dtype=float)
        expected = (len(self), len(self.supercell), 3)
        if values.shape != expected:
            raise ValueError(f"forces must have shape {expected}, got {values.shape}")
        block = len(self.stencil.signs)
        result: dict[DisplacementKey, np.ndarray] = {}
        for begin in range(0, len(self), block):
            key = self.configurations[begin].key
            # IFC_n = -d^(n-1) F / du^(n-1).
            result[key] = -self.stencil.contract(values[begin : begin + block])
        return result


def build_displacement_plan(
    supercell: Atoms,
    orbit_space: OrbitSpace,
    *,
    displacement: float,
) -> DisplacementPlan:
    stencil = CentralDifferenceStencil.for_force_constant(orbit_space.order, displacement)
    configurations: list[DisplacementConfiguration] = []
    for key in orbit_space.displacement_keys:
        for signs in stencil.signs:
            delta = np.zeros((len(supercell), 3), dtype=float)
            for sign, (atom, direction) in zip(signs, key, strict=True):
                delta[atom, direction] += sign * displacement
            configurations.append(DisplacementConfiguration(key, signs.copy(), delta))
    return DisplacementPlan(supercell.copy(), tuple(configurations), stencil)
