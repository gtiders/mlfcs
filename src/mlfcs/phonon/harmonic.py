"""Harmonic dynamical matrices and phonon frequencies in the positional Fourier gauge."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from ase.cell import Cell
from ase.dft.kpoints import BandPath, parse_path_string

from mlfcs.force_constants import ForceConstants
from mlfcs.foundation.arrays import readonly
from mlfcs.foundation.log import get_logger
from mlfcs.phonon.dynamics import (
    THZ_PER_SQRT_EV_PER_A2_AMU,
    accumulate_dynamical_matrices,
    prepare_dynamical_terms,
)
from mlfcs.phonon.grid import QStars, as_qgrid, mass_preserving_symmetry

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HarmonicBandResult:
    """Phonon frequencies and plotting metadata along a reciprocal-space path.

    ``qpoints`` are fractional reciprocal coordinates. ``frequencies_thz``
    contains signed ordinary frequencies in THz; negative values represent
    imaginary modes. Modes are sorted independently at each point, without
    branch tracking.

    ``distances`` gives cumulative path length in inverse angstrom.
    ``segments`` identifies connected path legs, while ``tick_positions`` and
    ``tick_labels`` define the labelled plotting axis.
    """

    path: str
    qpoints: np.ndarray
    frequencies_thz: np.ndarray
    distances: np.ndarray
    tick_positions: np.ndarray
    tick_labels: tuple[str, ...]
    segments: tuple[slice, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the band-path result arrays."""
        for name in ("qpoints", "frequencies_thz", "distances", "tick_positions"):
            object.__setattr__(self, name, readonly(getattr(self, name), np.float64))


def _band_path_axis(path: BandPath):
    """Construct cumulative reciprocal-space distance and labelled path segments."""
    points = np.asarray(path.kpts, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("band path requires sampled fractional reciprocal points")
    if not np.all(np.isfinite(points)):
        raise ValueError("band path q points must be finite")
    legs = parse_path_string(path.path)
    if not legs or any(len(names) < 2 for names in legs):
        raise ValueError("each connected band path must contain at least two vertices")
    distances = np.zeros(len(points))
    reciprocal = 2 * np.pi * np.asarray(path.cell.reciprocal())
    segments, tick_positions, tick_labels = [], [], []
    cursor, last_endpoint, distance = 0, -1, 0.0
    for names in legs:
        vertices = []
        for name in names:
            target = np.asarray(path.special_points[name])
            matches = np.flatnonzero(np.linalg.norm(points[cursor:] - target, axis=1) < 1e-8)
            if not len(matches):
                raise ValueError(f"band path is missing the ordered vertex {name!r}")
            index = cursor + int(matches[0])
            if vertices:
                start = vertices[-1]
                increments = np.linalg.norm(
                    np.diff(points[start : index + 1], axis=0) @ reciprocal, axis=1
                )
                distances[start : index + 1] = distance + np.concatenate(
                    ([0.0], np.cumsum(increments))
                )
                distance = float(distances[index])
                segments.append(slice(start, index + 1))
            else:
                if index != last_endpoint + 1:
                    raise ValueError("band path contains samples outside its labelled legs")
                distances[index] = distance
            vertices.append(index)
            cursor = index + 1
            label = "Γ" if name == "G" else name
            if tick_positions and distance == tick_positions[-1]:
                if label not in tick_labels[-1].split("|"):
                    tick_labels[-1] += "|" + label
            else:
                tick_positions.append(distance)
                tick_labels.append(label)
        last_endpoint = vertices[-1]
    if cursor != len(points):
        raise ValueError("band path contains samples after its final vertex")
    return distances, np.asarray(tick_positions), tuple(tick_labels), tuple(segments)


@dataclass(frozen=True, slots=True)
class HarmonicMeshResult:
    """Irreducible harmonic frequencies on a symmetry-reduced reciprocal grid.

    ``qpoints`` contains irreducible fractional reciprocal coordinates and
    ``weights`` gives the number of full-grid points represented by each one.
    ``frequencies_thz`` contains signed ordinary frequencies; negative values
    denote imaginary modes. ``mesh_matrix`` identifies the integer row lattice.
    ``full_frequencies()`` expands to the complete lexicographic QGrid order.
    """

    frequencies_thz: np.ndarray
    _stars: QStars = field(repr=False)
    qpoints: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    mesh_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Validate and freeze the irreducible mesh result."""
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
        """Expand irreducible frequencies to the full reciprocal-grid order."""
        return self._stars.expand(self.frequencies_thz)


class Harmonic:
    """Harmonic lattice dynamics constructed from primitive-cell FC2.

    The model Fourier transforms second-order force constants into
    mass-weighted dynamical matrices in the positional gauge,

        D[iα,jβ](q) = sum_R Phi[iα,jβ](R)
            * exp(2*pi*i*q·(R + s_j - s_i)) / sqrt(m_i*m_j).

    It evaluates dynamical matrices, signed phonon frequencies, irreducible
    reciprocal meshes, and labelled band paths. Atomic masses come from the
    associated ``ClusterSpace``.
    """

    def __init__(self, model: ForceConstants) -> None:
        """Construct harmonic lattice dynamics from a force-constant model containing FC2."""
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
        """Primitive-site masses in atomic mass units."""
        return self._masses

    def dynamical_matrices(self, qpoints: object) -> np.ndarray:
        """Evaluate mass-weighted dynamical matrices at reciprocal-space points.

        ``qpoints`` may be one fractional reciprocal coordinate, an array of
        coordinates, or ``QStars``. A star request evaluates its irreducible
        representatives; the stars must use symmetry that preserves this
        model's masses. The positional Fourier convention is used. Matrix
        elements have units eV/(angstrom**2 * atomic_mass).

        Return shape ``(3*N, 3*N)`` for one point or
        ``(nq, 3*N, 3*N)`` for a point batch or ``QStars``.
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
        """Return signed ordinary phonon frequencies in THz.

        Each eigenvalue ``lambda`` of the mass-weighted dynamical matrix gives
        ``sign(lambda) * sqrt(abs(lambda))`` with the unit conversion to THz.
        The conversion includes ``1/(2*pi)``, so these are ordinary rather than
        angular frequencies. Negative values represent imaginary modes and are
        not clipped to zero. Modes are sorted by eigenvalue independently at
        each q point.
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
        """Evaluate signed phonon frequencies on an irreducible reciprocal mesh.

        ``mesh`` may be three positive mesh sizes, a nonsingular integer
        supercell matrix, or an existing ``QGrid``. Reduction uses the subgroup
        preserving the primitive mass assignment and optionally includes time
        reversal. The result stores irreducible frequencies and integer star
        weights; full-grid frequencies are expanded on demand.
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

    def run_band_path(
        self, path: str | BandPath | None = None, *, npoints: int | None = None
    ) -> HarmonicBandResult:
        """Evaluate phonon frequencies along a labelled reciprocal-space path.

        If ``path`` is omitted, ASE selects the default path for the primitive
        Bravais lattice. A string such as ``"GX,LG"`` selects named points,
        with commas separating disconnected pieces. ``npoints`` sets the target
        sampling count over the complete path.

        A sampled ASE ``BandPath`` may be supplied when it uses the model's
        primitive-cell basis. The result includes fractional q points, signed
        frequencies, path distances, connected segments, and plotting ticks.
        This method does not apply a non-analytical long-range correction.
        """
        cell = Cell(self.model.cluster_space.cell)
        if isinstance(path, BandPath):
            if npoints is not None:
                raise ValueError("npoints cannot be supplied with a sampled BandPath")
            if not np.allclose(path.cell, cell, rtol=1e-10, atol=1e-10):
                raise ValueError("BandPath must use the model's primitive cell basis")
            prepared = path
        else:
            if path is not None and not isinstance(path, str):
                raise TypeError("path must be a string, ASE BandPath, or None")
            count = 200 if npoints is None else npoints
            if (
                isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer))
                or count < 2
            ):
                raise ValueError("npoints must be an integer of at least two")
            prepared = cell.bandpath(path, npoints=int(count))
        distances, ticks, labels, segments = _band_path_axis(prepared)
        return HarmonicBandResult(
            prepared.path,
            prepared.kpts,
            self.frequencies(prepared.kpts),
            distances,
            ticks,
            labels,
            segments,
        )


__all__ = ["Harmonic", "HarmonicBandResult", "HarmonicMeshResult"]
