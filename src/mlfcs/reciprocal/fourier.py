r"""Fourier kernels for phonon dynamical matrices, in one documented gauge.

Gauge
=====

The project uses the *positional* gauge for every dynamical matrix:

.. math::

    D_{ab}(q) = \frac{1}{\sqrt{m_a m_b}} \sum_R
        \Phi_{ab}(0, Rb) \exp\left[2 \pi i\, q \cdot (R + \tau_b - \tau_a)\right],

with the row-vector convention of the rest of the repository (a cell's rows are lattice
vectors, ``fractional @ cell`` is Cartesian, and ``q`` is a fractional reciprocal row
vector).  Two consequences matter:

* the phase carries the atomic positions, so an eigenvector component belongs to one atom
  of the primitive cell with its position already folded in.  The real-space Bloch field
  is therefore reconstructed as
  :math:`u_{aR} = N^{-1/2} \sum_q \exp[2 \pi i\, q \cdot (R + \tau_a)] \tilde u_a(q)`;
* :mod:`mlfcs.reciprocal.scph` and :mod:`mlfcs.reciprocal.sampling` share the gauge, so
  one symmetry representation :math:`U_g(q)` and one expansion of physical quantities
  serve both.  The former implementation used ``exp[2 pi i q . R]`` in the sampler, whose
  eigenvalues agree but whose eigenvectors differ by a per-atom phase; mixing the two
  would silently break every multi-atom expansion.

Two kernels build the same matrix from two equivalent descriptions of the force
constants: :func:`dynamical_matrix` from exact primitive-lattice tensors and
:func:`compact_dynamical_matrix` from a supercell-compact FC2 together with the periodic
index of its atoms.  ``tests/test_reciprocal_representation.py`` pins that they agree.

Nothing here knows about symmetry reduction; the kernels take the q points they are given.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
from ase import Atoms


class FourierTerm(NamedTuple):
    """One exact primitive-lattice FC2 term: ``Phi_ab(0, R)`` with its phase vector.

    ``images`` is the Cartesian-free fractional vector ``R + tau_b - tau_a`` whose dot
    product with ``q`` is the phase argument in the positional gauge.
    """

    first: int
    second: int
    images: np.ndarray
    tensor: np.ndarray


def fourier_terms(
    lattice: Mapping[tuple[int, int, tuple[int, int, int]], np.ndarray],
    primitive: Atoms,
) -> tuple[FourierTerm, ...]:
    """Build the phase vector and tensor of every exact primitive-lattice FC2 term."""
    scaled = primitive.get_scaled_positions(wrap=False)
    terms = []
    for (first, site, translation), tensor in lattice.items():
        vector = scaled[site] - scaled[first] + np.asarray(translation, dtype=float)
        terms.append(FourierTerm(int(first), int(site), vector, np.asarray(tensor, dtype=float)))
    return tuple(terms)


def _fold(matrix: np.ndarray) -> np.ndarray:
    """Return the Hermitian part of one (possibly batched) dynamical matrix."""
    return (matrix + matrix.conj().swapaxes(-1, -2)) / 2


def dynamical_matrix(
    terms: tuple[FourierTerm, ...], masses: np.ndarray, qpoint: np.ndarray
) -> np.ndarray:
    """Return the ``3n x 3n`` mass-weighted dynamical matrix at one q point."""
    n = len(masses)
    matrix = np.zeros((3 * n, 3 * n), dtype=complex)
    for term in terms:
        phase = np.exp(2j * np.pi * float(term.images @ np.asarray(qpoint, dtype=float)))
        matrix[3 * term.first : 3 * term.first + 3, 3 * term.second : 3 * term.second + 3] += (
            term.tensor * phase / np.sqrt(masses[term.first] * masses[term.second])
        )
    return _fold(matrix)


def dynamical_matrices(
    terms: tuple[FourierTerm, ...], masses: np.ndarray, qpoints: np.ndarray
) -> np.ndarray:
    """Return the dynamical matrices of a whole q batch in one bounded pass."""
    values = np.asarray(qpoints, dtype=float).reshape((-1, 3))
    n = len(masses)
    matrix = np.zeros((len(values), 3 * n, 3 * n), dtype=complex)
    for term in terms:
        phase = np.exp(2j * np.pi * (values @ term.images))
        matrix[:, 3 * term.first : 3 * term.first + 3, 3 * term.second : 3 * term.second + 3] += (
            phase[:, None, None] * term.tensor / np.sqrt(masses[term.first] * masses[term.second])
        )
    return _fold(matrix)


def compact_dynamical_matrix(
    compact: np.ndarray,
    cell_atoms: np.ndarray,
    cell_translations: np.ndarray,
    positions: np.ndarray,
    masses: np.ndarray,
    qpoint: np.ndarray,
) -> np.ndarray:
    """Return the dynamical matrix of a supercell-compact FC2 in the same gauge.

    ``compact`` is ``(n_primitive, n_supercell, 3, 3)``, ``cell_atoms`` is
    ``(n_cells, n_primitive)`` with the supercell atom index of each primitive site in
    each cell, ``cell_translations`` is ``(n_cells, 3)`` and ``positions`` the primitive
    fractional coordinates.  Summing the compact images over the cells is the same
    lattice sum as :func:`dynamical_matrix`, evaluated at q points commensurate with the
    supercell.
    """
    n_primitive = len(masses)
    values = compact[:, np.asarray(cell_atoms, dtype=np.int64).reshape(-1)].reshape(
        n_primitive, len(cell_translations), n_primitive, 3, 3
    )
    images = (
        np.asarray(cell_translations, dtype=float)[:, None, None, :]
        + positions[None, None, :, :]
        - positions[None, :, None, :]
    )
    phase = np.exp(2j * np.pi * np.einsum("cabd,d->cab", images, np.asarray(qpoint, dtype=float)))
    blocks = np.einsum("cab,acbuv->aubv", phase, values, optimize=True)
    mass = np.sqrt(masses[:, None] * masses[None, :])
    blocks /= mass[:, None, :, None]
    return _fold(blocks.reshape(3 * n_primitive, 3 * n_primitive))


__all__ = [
    "FourierTerm",
    "compact_dynamical_matrix",
    "dynamical_matrices",
    "dynamical_matrix",
    "fourier_terms",
]
