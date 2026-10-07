"""Screened dipole Ewald FC2 with a local symmetry-compatible correction.

Three-dimensional Ewald summation evaluates the long-range dipole interaction.
A finite local FC2 correction restores the required symmetry conditions within
the cluster-space parameterization without truncating the dipole tail. The
range-separation construction follows Zhou et al., Phys. Rev. B 100, 184309
(2019), Sec. II.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import erfc, exp, pi, sqrt

import numpy as np
from ase import units
from numba import njit
from scipy.linalg import lstsq

from mlfcs.cluster_space import ClusterSpace
from mlfcs.force_constants import CompactForceConstants
from mlfcs.foundation.arrays import as_int64_array, require_allocation
from mlfcs.foundation.errors import ConstraintProjectionError
from mlfcs.foundation.tensors import rotate_basis
from mlfcs.geometry.primitive import LatticeSite
from mlfcs.mapping import ClusterMap

# ASE defines this conversion from its own consistent physical constants.
_COULOMB = units.Hartree * units.Bohr


def _freeze(values):
    """Return an independent readonly float64 array."""
    values = np.array(values, dtype=np.float64, copy=True, order="C")
    values.setflags(write=False)
    return values


def _validate_inputs(space, born, dielectric, rtol, atol):
    """Validate Born charges and the electronic dielectric tensor.

    Born charges must be neutral and transform consistently under the primitive
    space group. The dielectric tensor must be symmetric, positive definite,
    and invariant under the same Cartesian symmetry operations.
    """
    if not isinstance(space, ClusterSpace):
        raise TypeError("cluster_space must be a ClusterSpace")
    if 2 not in space.orders:
        raise ValueError("Ewald local correction requires a cluster space containing FC2")
    born = _freeze(born)
    dielectric = _freeze(dielectric)
    if born.shape != (space.n_atoms, 3, 3) or not np.all(np.isfinite(born)):
        raise ValueError("born_charges must be finite with shape (primitive_atoms, 3, 3)")
    if dielectric.shape != (3, 3) or not np.all(np.isfinite(dielectric)):
        raise ValueError("dielectric must be finite with shape (3, 3)")
    if not np.isfinite(rtol) or not 0 < rtol < 1 or not np.isfinite(atol) or atol <= 0:
        raise ValueError("require 0 < rtol < 1 and positive finite atol in eV/angstrom**2")
    tolerance = 1e-8
    if not np.array_equal(dielectric, dielectric.T):
        raise ValueError("dielectric must be symmetric")
    if np.min(np.linalg.eigvalsh(dielectric)) <= 0:
        raise ValueError("dielectric must be positive definite")
    charge_scale = max(1.0, float(np.max(np.abs(born))))
    if np.max(np.abs(born.sum(axis=0))) > tolerance * charge_scale:
        raise ValueError("born_charges must satisfy charge neutrality; inputs are not corrected")
    for rotation, permutation in zip(
        space.symmetry.cartesian_rotations, space.symmetry.site_permutations, strict=True
    ):
        # Stored Cartesian rotations act on row vectors; tensors below use columns.
        transformed = np.einsum("ab,ibc,cd->iad", rotation.T, born, rotation)
        if not np.allclose(transformed, born[permutation], rtol=tolerance, atol=tolerance):
            raise ValueError("born_charges are inconsistent with primitive symmetry")
        if not np.allclose(
            rotation.T @ dielectric @ rotation, dielectric, rtol=tolerance, atol=tolerance
        ):
            raise ValueError("dielectric is inconsistent with primitive symmetry")
    return born, dielectric


def _vectors(cell, transform, radius, span=0.0):
    """Enumerate translations inside a sphere in the transformed metric."""
    transformed_cell = cell @ transform
    bounds = np.ceil(
        (radius + span) * np.linalg.norm(np.linalg.inv(transformed_cell), axis=0)
    ).astype(object)
    bounds = [int(value) for value in bounds]
    shape = (int(np.prod([2 * b + 1 for b in bounds], dtype=object)), 3)
    require_allocation("Ewald lattice vectors", shape)
    labels = np.array(list(product(*(range(-b, b + 1) for b in bounds))), dtype=np.int64)
    vectors = labels @ cell
    metric_vectors = vectors @ transform
    keep = np.einsum("ij,ij->i", metric_vectors, metric_vectors) <= (radius + span) ** 2 * (
        1 + 1e-14
    )
    return np.ascontiguousarray(vectors[keep])


@njit(cache=True)
def _sum_dipoles(
    positions,
    anchors,
    site_ids,
    born,
    inverse_epsilon,
    determinant_root,
    real_vectors,
    reciprocal_vectors,
    alpha,
    real_radius,
    volume,
    epsilon,
):
    """Evaluate the screened dipole Hessian from real, reciprocal, and self terms.

    The resulting Green-function Hessian is contracted with the Born effective
    charges to produce Cartesian FC2 blocks for the requested site pairs.
    """
    result = np.zeros((len(anchors), len(positions), 3, 3))
    for i in range(len(anchors)):
        for j in range(len(positions)):
            separation = positions[j] - positions[anchors[i]]
            dipole = np.zeros((3, 3))
            for shift in real_vectors:
                distance = separation + shift
                conjugate = inverse_epsilon @ distance
                squared = np.dot(distance, conjugate)
                if squared <= 1e-24 or squared > real_radius * real_radius:
                    continue
                rho = sqrt(squared)
                gaussian = exp(-alpha * alpha * squared)
                complement = erfc(alpha * rho)
                isotropic = complement / rho**3 + 2 * alpha / sqrt(pi) * gaussian / squared
                directional = (
                    3 * complement / rho**5
                    + 6 * alpha / sqrt(pi) * gaussian / rho**4
                    + 4 * alpha**3 / sqrt(pi) * gaussian / squared
                )
                for a in range(3):
                    for b in range(3):
                        dipole[a, b] += (
                            inverse_epsilon[a, b] * isotropic
                            - conjugate[a] * conjugate[b] * directional
                        ) / determinant_root
            for reciprocal in reciprocal_vectors:
                denominator = np.dot(reciprocal, epsilon @ reciprocal)
                phase = np.dot(reciprocal, separation)
                factor = (
                    4
                    * pi
                    / volume
                    * exp(-denominator / (4 * alpha * alpha))
                    / denominator
                    * np.cos(phase)
                )
                for a in range(3):
                    for b in range(3):
                        dipole[a, b] += factor * reciprocal[a] * reciprocal[b]
            if anchors[i] == j:
                dipole -= 4 * alpha**3 / (3 * sqrt(pi) * determinant_root) * inverse_epsilon
            result[i, j] = _COULOMB * born[site_ids[anchors[i]]].T @ dipole @ born[site_ids[j]]
    return result


def _raw_fc2(cell, positions, anchors, site_ids, born, epsilon, alpha, extent):
    """Evaluate uncorrected periodic dipole FC2 for finite Ewald cutoffs."""
    eigenvalues, eigenvectors = np.linalg.eigh(epsilon)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    real_radius = extent / alpha
    transformed = positions @ inverse_root
    span = float(np.linalg.norm(np.ptp(transformed, axis=0)))
    real = _vectors(cell, inverse_root, real_radius, span)
    reciprocal_cell = 2 * pi * np.linalg.inv(cell).T
    reciprocal = _vectors(reciprocal_cell, root, 2 * alpha * extent)
    reciprocal = np.ascontiguousarray(reciprocal[np.any(reciprocal != 0, axis=1)])
    require_allocation("Ewald compact FC2", (len(anchors), len(positions), 3, 3))
    values = _sum_dipoles(
        np.ascontiguousarray(positions),
        np.asarray(anchors, dtype=np.int64),
        np.asarray(site_ids, dtype=np.int64),
        born,
        np.linalg.inv(epsilon),
        sqrt(np.linalg.det(epsilon)),
        real,
        np.ascontiguousarray(reciprocal),
        alpha,
        real_radius,
        abs(np.linalg.det(cell)),
        epsilon,
    )
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("Ewald summation produced nonfinite force constants")
    return values


def _converged_fc2(cell, positions, anchors, sites, born, epsilon, alpha, rtol, atol):
    """Increase both summation radii until two successive FC2 changes meet tolerance."""
    extent = sqrt(-np.log(rtol)) + 1
    previous = None
    consecutive = 0
    for iteration in range(12):
        values = _raw_fc2(cell, positions, anchors, sites, born, epsilon, alpha, extent + iteration)
        if previous is not None:
            change = float(np.max(np.abs(values - previous), initial=0))
            scale = float(np.max(np.abs(values), initial=0))
            consecutive = consecutive + 1 if change <= atol + rtol * scale else 0
            if consecutive == 2:
                return values
        previous = values
    raise RuntimeError("Ewald FC2 did not converge after expanding both summation radii")


def _antisymmetric(values):
    """Extract xy, xz and yz antisymmetric components in primitive-site order."""
    return np.stack(
        (
            values[:, 0, 1] - values[:, 1, 0],
            values[:, 0, 2] - values[:, 2, 0],
            values[:, 1, 2] - values[:, 2, 1],
        ),
        axis=1,
    ).reshape(-1)


def _local_correction(space, raw_rows, rtol, atol):
    """Construct the minimum-norm local FC2 correction for onsite symmetry.

    The correction uses non-onsite FC2 orbit parameters already present in the
    cluster space. It cancels the antisymmetric part of the raw dipole row sums
    while minimizing the Frobenius norm of the expanded Cartesian correction.
    The onsite term is then fixed by the acoustic sum rule.
    """
    records = []
    columns = []
    whiteners = []
    for orbit in space.orbits[space.block(2).orbits]:
        if orbit.representative.body_order == 1 or orbit.dimension == 0:
            continue
        row_basis = np.zeros((space.n_atoms, 3, 3, orbit.dimension))
        metric = np.zeros((orbit.dimension, orbit.dimension))
        images = []
        for image, cluster in enumerate(orbit.clusters):
            rotation = space.symmetry.cartesian_rotations[orbit.operations[image]].T
            basis = rotate_basis(orbit.component_basis, rotation, orbit.permutations[image])
            shaped = basis.reshape(3, 3, orbit.dimension)
            row_basis[cluster.sites[0].site] += shaped
            metric += basis.T @ basis
            images.append((cluster, shaped))
        # Whitening makes the Euclidean coordinate norm equal the expanded-tensor norm.
        whitener = np.linalg.solve(np.linalg.cholesky(metric).T, np.eye(orbit.dimension))
        antisymmetric = np.stack(
            (
                row_basis[:, 0, 1] - row_basis[:, 1, 0],
                row_basis[:, 0, 2] - row_basis[:, 2, 0],
                row_basis[:, 1, 2] - row_basis[:, 2, 1],
            ),
            axis=1,
        )
        columns.append(antisymmetric.reshape(3 * space.n_atoms, orbit.dimension) @ whitener)
        whiteners.append(whitener)
        records.append(images)
    target = -_antisymmetric(raw_rows)
    tolerance = atol + rtol * float(np.max(np.abs(raw_rows), initial=0))
    matrix = np.concatenate(columns, axis=1) if columns else np.zeros((len(target), 0))
    if matrix.shape[1]:
        coordinates = lstsq(matrix, target, lapack_driver="gelsd")[0]
    else:
        coordinates = np.empty(0)
    if np.max(np.abs(matrix @ coordinates - target), initial=0) > tolerance:
        raise ConstraintProjectionError(
            "FC2 cutoff provides insufficient non-onsite space for the Ewald local correction"
        )
    result = []
    correction_rows = np.zeros_like(raw_rows)
    offset = 0
    for images, whitener in zip(records, whiteners, strict=True):
        size = whitener.shape[0]
        parameters = whitener @ coordinates[offset : offset + size]
        offset += size
        for cluster, basis in images:
            tensor = basis @ parameters
            correction_rows[cluster.sites[0].site] += tensor
            result.append((cluster, _freeze(tensor)))
    onsite = -(raw_rows + correction_rows)
    if np.max(np.abs(onsite - onsite.transpose(0, 2, 1)), initial=0) > tolerance:
        raise ConstraintProjectionError("Ewald local correction failed onsite symmetry")
    return tuple(result), onsite, float(np.linalg.norm(coordinates))


def _supercell_tensors(mapping, born, epsilon, alpha, correction, onsite, rtol, atol):
    """Realize corrected dipole FC2 in a periodic supercell and check its constraints.

    The infinite-range Ewald interaction is folded into the selected supercell;
    local corrections and the onsite acoustic-sum-rule term are then added.
    """
    space = mapping.cluster_space
    anchors = np.array(
        [mapping.atom_index(LatticeSite(i, (0, 0, 0))) for i in range(space.n_atoms)]
    )
    compact = _converged_fc2(
        mapping.cell,
        mapping.supercell_atoms.positions,
        anchors,
        mapping.primitive_site_indices,
        born,
        epsilon,
        alpha,
        rtol,
        atol,
    )
    for cluster, tensor in correction:
        target = mapping.atom_index(cluster.sites[1])
        compact[cluster.sites[0].site, target] += tensor
    for site, anchor in enumerate(anchors):
        compact[site, anchor] += onsite[site]
    scale = float(np.max(np.abs(compact), initial=0))
    asr = float(np.max(np.abs(compact.sum(axis=1)), initial=0))
    if asr > atol + rtol * scale:
        raise ConstraintProjectionError("corrected Ewald supercell FC2 failed ASR")
    anchor_labels = np.column_stack(
        (np.arange(space.n_atoms), np.zeros((space.n_atoms, 3), dtype=np.int64))
    )
    for atom, site in enumerate(mapping.primitive_site_indices):
        translation = as_int64_array(
            [[-int(value) for value in mapping.lattice_translations[atom]]],
            name="Ewald inverse translation",
        )
        counterparts = mapping.map_labels(anchor_labels, translation)[:, 0]
        difference = compact[:, atom] - compact[site, counterparts].transpose(0, 2, 1)
        if np.max(np.abs(difference), initial=0) > atol + rtol * scale:
            raise ConstraintProjectionError("corrected Ewald FC2 failed exchange symmetry")
    return CompactForceConstants({2: _freeze(compact)}, mapping), asr


@dataclass(frozen=True, slots=True, init=False)
class DipoleEwald:
    """Long-range harmonic dipole model for a fixed periodic supercell.

    The model evaluates screened dipole-dipole FC2 from primitive Born
    effective charges and the electronic dielectric tensor using three-
    dimensional Ewald summation. Omitting the reciprocal zero vector imposes
    zero macroscopic electric field. Born charges use primitive-site order,
    polarization then displacement axes, and elementary-charge units. The
    dielectric tensor is expressed in Cartesian coordinates.

    A finite local correction is constructed from non-onsite FC2 orbits in the
    associated ``ClusterSpace``. Its coefficients minimize the Frobenius norm
    of the expanded Cartesian correction while restoring onsite symmetry; the
    onsite term then enforces the acoustic sum rule. The correction support
    depends on the FC2 cutoff, but the infinite dipole tail is not truncated.
    Exchange and space-group symmetry are preserved; Born-Huang rotational
    constraints are not imposed.

    The corrected interaction is folded into the selected supercell and stored
    as compact FC2. It can evaluate harmonic forces and be exported for that
    supercell, but the folded result alone does not define arbitrary-q long-range
    response. Born charges, dielectric response, and interaction centers remain
    fixed at the reference geometry, so this is not a nonlinear point-charge
    potential. A different supercell requires another model.

    ``rtol`` and ``atol`` control summation convergence and residual checks;
    ``atol`` is in eV/angstrom**2. Successive cutoff expansions are a numerical
    convergence check, not a rigorous tail certificate. Insufficient local
    correction support raises ``ConstraintProjectionError``. ``correction_norm``
    is the Frobenius norm of the expanded local correction, and
    ``asr_residual`` is the maximum absolute row-sum residual of the folded FC2.
    """

    cluster_map: ClusterMap
    born_charges: np.ndarray
    dielectric: np.ndarray
    correction_norm: float
    asr_residual: float
    rtol: float
    atol: float
    _tensors: CompactForceConstants

    def __init__(self, cluster_map, *, born_charges, dielectric, rtol=1e-8, atol=1e-10):
        """Construct corrected long-range FC2 for one mapped supercell.

        ``born_charges[i, alpha, beta]`` is the polarization response along
        Cartesian direction ``alpha`` to displacement along ``beta`` for
        primitive site ``i``. ``dielectric`` is the electronic dielectric
        tensor in the same Cartesian frame. ``rtol`` and ``atol`` set the Ewald
        convergence and residual tolerances; ``atol`` has FC2 units.
        """
        if not isinstance(cluster_map, ClusterMap):
            raise TypeError("cluster_map must be a ClusterMap")
        space = cluster_map.cluster_space
        born, epsilon = _validate_inputs(space, born_charges, dielectric, rtol, atol)
        screened_volume = abs(np.linalg.det(space.cell)) / sqrt(np.linalg.det(epsilon))
        alpha = sqrt(pi) / screened_volume ** (1 / 3)
        primitive = _converged_fc2(
            space.cell,
            space.cartesian_positions,
            np.arange(space.n_atoms),
            np.arange(space.n_atoms),
            born,
            epsilon,
            alpha,
            rtol,
            atol,
        )
        correction, onsite, norm = _local_correction(space, primitive.sum(axis=1), rtol, atol)
        tensors, asr = _supercell_tensors(
            cluster_map,
            born,
            epsilon,
            alpha,
            correction,
            onsite,
            rtol,
            atol,
        )
        for name, value in (
            ("cluster_map", cluster_map),
            ("born_charges", born),
            ("dielectric", epsilon),
            ("correction_norm", norm),
            ("asr_residual", asr),
            ("rtol", float(rtol)),
            ("atol", float(atol)),
            ("_tensors", tensors),
        ):
            object.__setattr__(self, name, value)

    def force_constants(self):
        """Return corrected long-range FC2 in compact supercell form.

        The Cartesian tensors are in eV/angstrom**2 and include periodic-image,
        self, local-correction, and onsite contributions. The result can be
        added to short-range FC2 or exported without repeating the Ewald sum.
        Because the interaction is folded into one finite supercell, this
        representation alone cannot recover arbitrary-q long-range response.
        """
        return self._tensors

    def forces(self, displacements):
        """Apply the long-range harmonic FC2 to Cartesian displacement frames.

        For displacements ``u`` in the mapped supercell atom order, return
        ``F = -Phi u``. A single frame has shape ``(n_atoms, 3)`` and a batch
        has shape ``(n_frames, n_atoms, 3)``. Inputs are in angstrom and forces
        are in eV/angstrom. The compact FC2 computed at initialization is reused.
        """
        return self._tensors.harmonic_forces(displacements)


__all__ = ["DipoleEwald"]
