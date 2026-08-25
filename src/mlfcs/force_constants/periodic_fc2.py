"""Source-supercell harmonic Hessians and exact-R FC2 complements."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy import sparse

from mlfcs.force_constants.dense import expand_compact_fc2
from mlfcs.interactions.realization import InteractionAliasingError
from mlfcs.interactions.tensors import TensorAction, _null_space_from_gram
from mlfcs.structure.integer_lattice import IntegerLatticeQuotient
from mlfcs.structure.relation import StructureRelation


def _source_fingerprint(relation: StructureRelation) -> str:
    digest = sha256(b"mlfcs-periodic-fc2-source-v1")
    quotient = IntegerLatticeQuotient(relation.supercell_matrix)
    for values in (
        np.asarray(relation.primitive.numbers, dtype=np.int64),
        np.round(np.asarray(relation.primitive.cell, dtype=np.float64), decimals=12),
        np.round(
            np.asarray(relation.primitive.get_scaled_positions(wrap=True), dtype=np.float64),
            decimals=12,
        ),
        np.asarray(quotient.hnf, dtype=np.int64),
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _pair_offset(site: int, atom: int, n_reference: int) -> int:
    return (site * n_reference + atom) * 9


def _finite_pair_generators(frame):
    relation = frame.relation
    index = relation.index
    n_primitive = len(relation.primitive)
    n_reference = len(relation.reference)
    labels = tuple((site, atom) for site in range(n_primitive) for atom in range(n_reference))
    generators = []
    for permutation, cartesian in zip(
        frame.symmetry.atom_permutations,
        frame.symmetry.cartesian_rotations,
        strict=True,
    ):
        action = TensorAction(cartesian.T, (0, 1), 2).as_matrix()
        mapped = []
        for site, atom in labels:
            first = int(permutation[index.representative(site)])
            second = int(permutation[atom])
            second = index.translate_atom(second, -index.translations[first])
            mapped.append((int(index.primitive[first]), second))
        generators.append((tuple(mapped), action))
    transpose = TensorAction(np.eye(3), (1, 0), 2).as_matrix()
    mapped = []
    for site, atom in labels:
        mapped.append(
            (
                int(index.primitive[atom]),
                index.atom(site, -index.translations[atom]),
            )
        )
    generators.append((tuple(mapped), transpose))
    positions = {label: position for position, label in enumerate(labels)}
    return labels, tuple(
        (
            np.asarray([positions[label] for label in mapped], dtype=np.int32),
            action,
        )
        for mapped, action in generators
    )


def _finite_pair_basis(frame) -> sparse.csc_matrix:
    """Generate the symmetry-allowed compact finite FC2 basis by group orbits."""
    labels, generators = _finite_pair_generators(frame)
    n_reference = len(frame.relation.reference)
    unseen = set(range(len(labels)))
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    column_offset = 0
    identity = np.eye(9)
    while unseen:
        representative = min(unseen)
        transport = {representative: identity}
        queue = [representative]
        constraint_gram = np.zeros((9, 9))
        while queue:
            current = queue.pop(0)
            current_transport = transport[current]
            for mapping, action in generators:
                target = int(mapping[current])
                candidate = action @ current_transport
                previous = transport.get(target)
                if previous is None:
                    transport[target] = candidate
                    queue.append(target)
                else:
                    residual = candidate - previous
                    if np.linalg.norm(residual) > 1e-10:
                        constraint_gram += residual.T @ residual
        orbit = sorted(transport)
        unseen.difference_update(orbit)
        invariant = (
            _null_space_from_gram(constraint_gram, 1e-10)[0]
            if np.any(constraint_gram)
            else identity
        )
        if not invariant.shape[1]:
            raise RuntimeError("finite pair orbit has no invariant Cartesian tensor")
        invariant /= np.sqrt(len(orbit))
        for label_position in orbit:
            site, atom = labels[label_position]
            values = transport[label_position] @ invariant
            components, local_columns = np.nonzero(np.abs(values) > 1e-13)
            rows.extend(
                _pair_offset(site, atom, n_reference) + int(component)
                for component in components
            )
            columns.extend(column_offset + int(column) for column in local_columns)
            data.extend(
                float(values[component, column])
                for component, column in zip(components, local_columns, strict=True)
            )
        column_offset += invariant.shape[1]
    raw_dimension = len(labels) * 9
    return sparse.coo_matrix(
        (data, (rows, columns)), shape=(raw_dimension, column_offset)
    ).tocsc()


def _compact_asr(relation: StructureRelation) -> sparse.csr_matrix:
    n_primitive = len(relation.primitive)
    n_reference = len(relation.reference)
    rows = []
    columns = []
    data = []
    for site in range(n_primitive):
        for alpha in range(3):
            for beta in range(3):
                row = (site * 3 + alpha) * 3 + beta
                for atom in range(n_reference):
                    rows.append(row)
                    columns.append(_pair_offset(site, atom, n_reference) + alpha * 3 + beta)
                    data.append(1.0)
    return sparse.coo_matrix(
        (data, (rows, columns)),
        shape=(n_primitive * 9, n_primitive * n_reference * 9),
    ).tocsr()


def _dense_null_space(matrix: np.ndarray, tolerance: float = 1e-11) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.shape[0] == 0:
        return np.eye(values.shape[1])
    _left, singular, right = np.linalg.svd(values, full_matrices=True)
    threshold = (
        tolerance * max(values.shape) * float(singular[0]) if len(singular) else 0.0
    )
    rank = int(np.count_nonzero(singular > threshold))
    result = right[rank:].T
    residual = float(np.max(np.abs(values @ result))) if result.size else 0.0
    if residual > max(tolerance * 100, 1e-9):
        raise RuntimeError(f"periodic FC2 null-space residual is {residual:.6e}")
    return result


def _compact_exact_basis(calculation, relation: StructureRelation) -> np.ndarray:
    """Realize every primitive-orbit FC2 parameter in compact source coordinates."""
    from mlfcs.force_constants.expansion import expand_primitive_parameters

    dimension = sum(orbit.dimension for orbit in calculation.primitive_orbit_space.orbits)
    shape = (len(relation.primitive), len(relation.reference), 3, 3)
    result = np.zeros((int(np.prod(shape)), dimension))
    for column in range(dimension):
        parameters = np.zeros(dimension)
        parameters[column] = 1.0
        sparse_fc2 = expand_primitive_parameters(
            calculation.primitive_orbit_space, parameters
        )
        compact = np.zeros(shape)
        for sites, translations, tensor in zip(
            sparse_fc2.sites,
            sparse_fc2.translations,
            sparse_fc2.tensors,
            strict=True,
        ):
            atom = relation.index.atom(int(sites[1]), translations[0])
            compact[int(sites[0]), atom] += tensor
        result[:, column] = compact.reshape(-1)
    return result


def _exact_asr_constraints(orbit_space, tolerance: float = 1e-12) -> sparse.csr_matrix:
    """Build the FC2 exact-parameter ASR map without importing workflow layers."""
    dimensions = [orbit.dimension for orbit in orbit_space.orbits]
    offsets = np.cumsum([0, *dimensions])
    equations: dict[tuple[int, ...], int] = {}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for orbit_index, orbit in enumerate(orbit_space.orbits):
        representative = orbit.basis @ np.linalg.inv(orbit.basis[orbit.pivots])
        for image in orbit.images:
            image_basis = image.action.apply_columns(representative)
            for component in range(9):
                directions = np.unravel_index(component, (3, 3))
                key = tuple(image.key.labels[:-1]) + tuple(int(value) for value in directions)
                equation = equations.setdefault(key, len(equations))
                nonzero = np.flatnonzero(np.abs(image_basis[component]) > tolerance)
                rows.extend([equation] * len(nonzero))
                columns.extend(int(offsets[orbit_index] + value) for value in nonzero)
                data.extend(float(image_basis[component, value]) for value in nonzero)
    return sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(equations), int(offsets[-1]))
    ).tocsr()


@dataclass(frozen=True, slots=True)
class PeriodicFC2RankReport:
    raw_dimension: int
    translation_reduced_raw_dimension: int
    symmetry_reduced_dimension: int
    asr_dimension: int
    exact_dimension: int
    exact_rank: int
    completion_dimension: int
    hybrid_dimension: int
    numerical_rank: int
    discarded_dependent_directions: int
    rank_tolerance: float


@dataclass(frozen=True, slots=True)
class SupercellHessianSpace:
    """ASR- and symmetry-allowed periodic FC2 space for one source supercell."""

    relation: StructureRelation
    compact_basis: np.ndarray
    completion_basis: np.ndarray
    exact_map: np.ndarray
    exact_parameter_map: sparse.csc_matrix
    rank_report: PeriodicFC2RankReport

    @classmethod
    def build(cls, calculation, *, rank_tolerance: float | None = None):
        relation = calculation.frame.relation
        finite = _finite_pair_basis(calculation.frame)
        asr = _compact_asr(relation) @ finite
        asr_map = _dense_null_space(asr.toarray(), tolerance=1e-11)
        compact_basis = np.asarray(finite @ asr_map)

        exact_constraints = _exact_asr_constraints(calculation.primitive_orbit_space)
        exact_parameter_map = sparse.csc_matrix(
            _dense_null_space(exact_constraints.toarray(), tolerance=1e-11)
        )
        exact_raw = _compact_exact_basis(calculation, relation)
        exact_map = compact_basis.T @ exact_raw @ exact_parameter_map
        singular = np.linalg.svd(exact_map, compute_uv=False)
        default_tolerance = (
            np.finfo(float).eps
            * max(exact_map.shape)
            * (float(singular[0]) if len(singular) else 1.0)
        )
        tolerance = default_tolerance if rank_tolerance is None else float(rank_tolerance)
        if tolerance <= 0:
            raise ValueError("periodic FC2 rank tolerance must be positive")
        exact_rank = int(np.count_nonzero(singular > tolerance))
        if exact_rank != exact_map.shape[1]:
            raise InteractionAliasingError(
                "ASR-constrained exact FC2 realization is aliased in the source supercell"
            )
        left, _singular, _right = np.linalg.svd(exact_map, full_matrices=True)
        completion_coordinates = left[:, exact_rank:]
        completion_basis = compact_basis @ completion_coordinates
        hybrid_dimension = exact_rank + completion_basis.shape[1]
        rank_report = PeriodicFC2RankReport(
            raw_dimension=(3 * len(relation.reference))
            * (3 * len(relation.reference) + 1)
            // 2,
            translation_reduced_raw_dimension=compact_basis.shape[0],
            symmetry_reduced_dimension=finite.shape[1],
            asr_dimension=compact_basis.shape[1],
            exact_dimension=exact_map.shape[1],
            exact_rank=exact_rank,
            completion_dimension=completion_basis.shape[1],
            hybrid_dimension=hybrid_dimension,
            numerical_rank=hybrid_dimension,
            discarded_dependent_directions=finite.shape[1] - compact_basis.shape[1],
            rank_tolerance=tolerance,
        )
        return cls(
            relation,
            compact_basis,
            completion_basis,
            exact_map,
            exact_parameter_map,
            rank_report,
        )

    def full_completion_basis(self) -> np.ndarray:
        columns = []
        shape = (len(self.relation.primitive), len(self.relation.reference), 3, 3)
        for values in self.completion_basis.T:
            columns.append(
                expand_compact_fc2(
                    values.reshape(shape), self.relation.reference
                ).reshape(-1)
            )
        n = len(self.relation.reference) * 3
        return np.asarray(columns).T if columns else np.empty((n * n, 0))


@dataclass(frozen=True, slots=True)
class PeriodicFC2Completion:
    """A legitimate source-periodic harmonic Hessian outside the exact-R FC2 span."""

    relation: StructureRelation
    compact_hessian: np.ndarray
    rank_report: PeriodicFC2RankReport

    def __post_init__(self) -> None:
        values = np.asarray(self.compact_hessian, dtype=float)
        expected = (
            len(self.relation.primitive),
            len(self.relation.reference),
            3,
            3,
        )
        if values.shape != expected:
            raise ValueError(
                f"periodic FC2 compact Hessian must have shape {expected}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("periodic FC2 compact Hessian contains non-finite values")

    @property
    def source_fingerprint(self) -> str:
        return _source_fingerprint(self.relation)

    def full_hessian(self, reference=None) -> np.ndarray:
        target = self.relation.reference if reference is None else reference
        target_relation = StructureRelation.from_atoms(self.relation.primitive, target)
        source_q = IntegerLatticeQuotient(self.relation.supercell_matrix)
        target_q = IntegerLatticeQuotient(target_relation.supercell_matrix)
        if not np.array_equal(source_q.hnf, target_q.hnf):
            raise ValueError(
                "periodic FC2 completion belongs to a different source supercell"
            )
        source_full = expand_compact_fc2(
            self.compact_hessian, self.relation.reference
        )
        source_to_target = np.asarray(
            [
                target_relation.index.atom(int(site), translation)
                for site, translation in zip(
                    self.relation.primitive_index,
                    self.relation.cell_translation,
                    strict=True,
                )
            ],
            dtype=np.int32,
        )
        result = np.empty_like(source_full)
        result[np.ix_(source_to_target, source_to_target)] = source_full
        return result


__all__ = [
    "PeriodicFC2Completion",
    "PeriodicFC2RankReport",
    "SupercellHessianSpace",
]
