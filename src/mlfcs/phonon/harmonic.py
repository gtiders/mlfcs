"""Harmonic frequencies at requested q points in the positional Fourier gauge."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from mlfcs._arrays import readonly
from mlfcs.force_constants import ForceConstants
from mlfcs.log import get_logger
from mlfcs.phonon.dynamics import (
    THZ_PER_SQRT_EV_PER_A2_AMU,
    accumulate_dynamical_matrices,
    prepare_dynamical_terms,
)
from mlfcs.phonon.grid import QStars, as_qgrid, mass_preserving_symmetry

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HarmonicMeshResult:
    """Readonly irreducible harmonic frequencies with their sampling metadata.

    qpoints is (n_stars, 3) in fractional reciprocal coordinates; integer
    weights counts members of each star and sums to the full-grid size.
    frequencies_thz is (n_stars, 3*n_atoms), with negative entries for
    imaginary modes. mesh_matrix is the (3, 3) supercell row matrix.
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
        """Return a new (grid.size, 3*n_atoms) frequency array in full-grid order."""
        return self._stars.expand(self.frequencies_thz)


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
        terms, lattice = prepare_dynamical_terms(model)
        tensors = np.asarray(lattice.tensors, dtype=np.float64).reshape(-1, 3, 3)
        self.model = model
        self._masses = masses
        self._terms = terms
        self._tensors = tensors
        logger.info(
            "Harmonic preparation complete: terms=%d elapsed_s=%.2f",
            len(terms.first_sites),
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
        Invalid coordinates raise ValueError. Callers own star/model compatibility. Inputs and
        model coefficients remain unchanged.
        """
        if isinstance(qpoints, QStars):
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
        matrices = accumulate_dynamical_matrices(
            np.ascontiguousarray(points), self._terms, self._tensors, len(self.masses)
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
        frequencies = (
            np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * THZ_PER_SQRT_EV_PER_A2_AMU
        )
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
        or an existing grid. time_reversal includes q/-q pairing.
        Reduction uses the mass-preserving subgroup of the primitive symmetry.
        Return readonly qpoints, integer weights and signed THz frequencies;
        full-grid frequencies are allocated only on explicit expansion.
        """
        started = perf_counter()
        grid = as_qgrid(mesh)
        symmetry = mass_preserving_symmetry(self.model.cluster_space.symmetry, self.masses)
        logger.info(
            "Harmonic mesh started: qpoints=%d mass_preserving_operations=%d time_reversal=%s",
            grid.size,
            symmetry.size,
            time_reversal,
        )
        stars = QStars(
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
