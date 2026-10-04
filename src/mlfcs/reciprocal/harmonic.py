"""Harmonic frequencies at requested q points in the positional Fourier gauge."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numba import njit
from scipy.constants import angstrom, atomic_mass, electron_volt

from mlfcs._arrays import readonly
from mlfcs.core.log import get_logger
from mlfcs.force_constants import ForceConstants
from mlfcs.force_constants.lattice import expand
from mlfcs.reciprocal.grid import QStars, _grid, _mass_symmetry

_THZ = np.sqrt(electron_volt / (angstrom**2 * atomic_mass)) / (2.0 * np.pi * 1e12)
logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HarmonicMeshResult:
    """Readonly irreducible harmonic frequencies with their sampling metadata.

    qpoints is (n_stars, 3) in fractional reciprocal coordinates; integer
    weights counts members of each star and sums to the full-grid size.
    frequencies_thz is (n_stars, 3*n_atoms), with negative entries for
    imaginary modes. mesh_matrix is the exact (3, 3) row-cell matrix.
    full_frequencies() expands into lexicographic QGrid label order on demand.
    """

    frequencies_thz: np.ndarray
    _stars: QStars = field(repr=False)
    qpoints: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    mesh_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Capture independent readonly result arrays and check their leading shapes."""
        count = len(self._stars.representatives)
        points = readonly(self._stars.points, np.float64)
        values = readonly(self.frequencies_thz, np.float64)
        if points.shape != (count, 3) or values.shape != (
            count,
            3 * self._stars.symmetry.site_permutations.shape[1],
        ):
            raise ValueError("harmonic mesh result has inconsistent point/frequency shapes")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
            raise ValueError("harmonic mesh result must be finite")
        object.__setattr__(self, "qpoints", points)
        object.__setattr__(self, "frequencies_thz", values)
        object.__setattr__(self, "weights", self._stars.weights)
        object.__setattr__(self, "mesh_matrix", self._stars.grid.matrix)

    def full_frequencies(self) -> np.ndarray:
        """Return a new (grid.size, 3*n_atoms) frequency array in exact full-grid order."""
        return self._stars.expand(self.frequencies_thz)


def _operation_key(symmetry, operation):
    """Encode one complete affine operation for exact structural-subgroup membership."""
    return tuple(
        getattr(symmetry, field)[operation].tobytes()
        for field in (
            "rotations",
            "translations",
            "cartesian_rotations",
            "site_permutations",
            "site_shifts",
        )
    )


def internal_basis(masses: np.ndarray) -> np.ndarray:
    """Return a (3*N, 3*(N-1)) orthonormal complement of mass-weighted translations.

    masses is a positive per-site mass vector in atomic mass units. Atom-major
    Cartesian rows exclude the three Gamma translation directions. For one
    site return (3, 0); input masses are not modified.
    """
    count = len(masses)
    if count == 1:
        return np.empty((3, 0))
    weights = np.sqrt(masses)
    weights /= np.linalg.norm(weights)
    direction = weights.copy()
    direction[-1] -= 1.0
    direction /= np.linalg.norm(direction)
    householder = np.eye(count) - 2.0 * np.outer(direction, direction)
    basis = np.zeros((3 * count, 3 * (count - 1)))
    for axis in range(3):
        basis[axis::3, axis::3] = householder[:, :-1]
    return basis


@njit(cache=True, nogil=True)
def _dynamical(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    images: np.ndarray,
    tensors: np.ndarray,
    mass_weights: np.ndarray,
    n_sites: int,
) -> np.ndarray:
    """Fourier-accumulate requested q points and symmetrize Hermitian roundoff.

    points is fractional reciprocal (nq, 3); first/second index primitive sites,
    images holds fractional positional-gauge separations, tensors is (terms, 3, 3)
    in eV/angstrom**2, and mass_weights is 1/sqrt(m_i*m_j) in inverse atomic mass
    units. Return complex128 (nq, 3*N, 3*N), before conversion to THz.
    Uses exp(+2*pi*i*q.dot(image)); no full reciprocal grid is required.
    """
    matrices = np.zeros((len(points), 3 * n_sites, 3 * n_sites), dtype=np.complex128)
    for iq in range(len(points)):
        for term in range(len(first)):
            angle = 0.0
            for axis in range(3):
                angle += points[iq, axis] * images[term, axis]
            phase = 2.0 * np.pi * angle
            real = np.cos(phase) * mass_weights[term]
            imaginary = np.sin(phase) * mass_weights[term]
            row = 3 * first[term]
            column = 3 * second[term]
            for alpha in range(3):
                for beta in range(3):
                    value = tensors[term, alpha, beta]
                    matrices[iq, row + alpha, column + beta] += complex(
                        value * real, value * imaginary
                    )
        for row in range(3 * n_sites):
            for column in range(row, 3 * n_sites):
                value = 0.5 * (matrices[iq, row, column] + np.conj(matrices[iq, column, row]))
                matrices[iq, row, column] = value
                matrices[iq, column, row] = np.conj(value)
    return matrices


class Harmonic:
    """FC2 positional-gauge Fourier model with per-primitive-site masses.

    Parameters
    ----------
    model : ForceConstants
        Model containing FC2. Retained by reference; lattice tensors and Fourier
        metadata are derived during initialization.

    Notes
    -----
    Masses are readonly snapshots in atomic mass units. Explicit points allow
    arbitrary per-site masses; mesh reduction uses only mass-preserving
    structural operations. Fractional reciprocal row coordinates pair directly with
    fractional real-space separations in the Fourier phase. QStars requests
    compute only representatives; explicit points support arbitrary band paths.
    No calculator is called or supercell model created.
    """

    def __init__(self, model: ForceConstants) -> None:
        """Require FC2 and cache Fourier terms with the cluster space's readonly masses."""
        if not isinstance(model, ForceConstants) or 2 not in model.coefficients:
            raise ValueError("harmonic frequencies require ForceConstants containing FC2")
        started = perf_counter()
        logger.info("Harmonic preparation started: primitive_atoms=%d", model.cluster_space.n_atoms)
        space = model.cluster_space
        masses = space.masses
        lattice = expand(model, 2)
        first = np.asarray([sites[0] for sites in lattice.sites], dtype=np.int32)
        second = np.asarray([sites[1] for sites in lattice.sites], dtype=np.int32)
        images = np.asarray(
            [
                np.asarray(translations[0])
                + space.scaled_positions[sites[1]]
                - space.scaled_positions[sites[0]]
                for sites, translations in zip(lattice.sites, lattice.translations, strict=True)
            ],
            dtype=np.float64,
        ).reshape(-1, 3)
        tensors = np.asarray(lattice.tensors, dtype=np.float64).reshape(-1, 3, 3)
        mass_weights = 1.0 / np.sqrt(masses[first] * masses[second])
        self.model = model
        self._masses = masses
        self._first = first
        self._second = second
        self._images = images
        self._tensors = tensors
        self._mass_weights = mass_weights
        logger.info(
            "Harmonic preparation complete: terms=%d elapsed_s=%.2f",
            len(first),
            perf_counter() - started,
        )

    @property
    def masses(self) -> np.ndarray:
        """Readonly primitive-site mass snapshot in atomic mass units; no setter is provided."""
        return self._masses

    def dynamical_matrices(self, qpoints: object) -> np.ndarray:
        """Return complex dynamical matrices at fractional reciprocal points.

        qpoints is (3,), (nq, 3), or QStars using a mass-preserving subgroup
        of this model's primitive symmetry. Empty point batches are refused.
        Return (3*N, 3*N) for one point, otherwise (nq, 3*N, 3*N), with QStars
        using only representatives. Entries have units eV/(angstrom**2*atomic_mass).
        Invalid coordinates or inconsistent symmetry raise ValueError. Inputs and
        model coefficients remain unchanged.
        """
        if isinstance(qpoints, QStars):
            symmetry = self.model.cluster_space.symmetry
            other = qpoints.symmetry
            keys = {_operation_key(symmetry, i) for i in range(symmetry.size)}
            if symmetry.symprec != other.symprec or any(
                _operation_key(other, i) not in keys for i in range(other.size)
            ):
                raise ValueError("q stars use a different primitive symmetry from FC2")
            if not np.all(self.masses[other.site_permutations] == self.masses[None, :]):
                raise ValueError("q stars contain operations exchanging different masses")
            points = qpoints.points
            single = False
        else:
            values = np.asarray(qpoints, dtype=np.float64)
            single = values.shape == (3,)
            points = values.reshape(1, 3) if single else values
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                "q points must be nonempty finite fractional coordinates of shape (n, 3)"
            )
        matrices = _dynamical(
            np.ascontiguousarray(points),
            self._first,
            self._second,
            self._images,
            self._tensors,
            self._mass_weights,
            len(self.masses),
        )
        return matrices[0] if single else matrices

    def frequencies(self, qpoints: object) -> np.ndarray:
        """Return signed ordinary frequencies in THz in ascending eigenvalue order.

        qpoints follows dynamical_matrices. Output is (3*N,) for one point, otherwise
        (nq, 3*N); QStars evaluates representatives only. Negative entries encode
        imaginary modes as -sqrt(abs(eigenvalue)); they are not clipped to zero.
        The conversion includes 1/(2*pi), so these are not angular frequencies.
        """
        started = perf_counter()
        matrices = self.dynamical_matrices(qpoints)
        eigenvalues = np.linalg.eigvalsh(matrices)
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * _THZ
        n_qpoints = len(frequencies) if frequencies.ndim == 2 else 1
        logger.info(
            "Harmonic frequencies: qpoints=%d modes_per_q=%d imaginary_modes=%d "
            "range=[%.8g, %.8g] THz elapsed=%.2f s",
            n_qpoints,
            frequencies.shape[-1],
            int(np.count_nonzero(frequencies < 0.0)),
            float(np.min(frequencies)),
            float(np.max(frequencies)),
            perf_counter() - started,
        )
        return frequencies

    def mesh(self, mesh: object, *, time_reversal: bool = True) -> HarmonicMeshResult:
        """Compute an irreducible mesh from positive sizes, an integer matrix or QGrid.

        mesh is (nx, ny, nz), a nonsingular integer (3, 3) row-cell matrix,
        or an existing exact grid. time_reversal includes q/-q pairing.
        Reduction uses the mass-preserving subgroup of the primitive symmetry.
        Return readonly qpoints, integer weights and signed THz frequencies;
        full-grid frequencies are allocated only on explicit expansion.
        """
        started = perf_counter()
        grid = _grid(mesh)
        symmetry = _mass_symmetry(self.model.cluster_space.symmetry, self.masses)
        logger.info(
            "Harmonic mesh started: qpoints=%d mass_preserving_operations=%d time_reversal=%s",
            grid.size,
            symmetry.size,
            time_reversal,
        )
        stars = QStars.from_symmetry(
            grid,
            symmetry,
            time_reversal=time_reversal,
        )
        result = HarmonicMeshResult(self.frequencies(stars), stars)
        logger.info(
            "Harmonic mesh complete: full_qpoints=%d irreducible_qpoints=%d elapsed_s=%.2f",
            grid.size,
            len(result.qpoints),
            perf_counter() - started,
        )
        return result


__all__ = ["Harmonic", "HarmonicMeshResult"]
