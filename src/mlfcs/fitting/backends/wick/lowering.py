"""Lower fitted Wick coordinates to canonical Taylor parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy import sparse
from scipy.linalg import qr, solve_triangular

from mlfcs.interactions.keys import InteractionKey


@dataclass(frozen=True, slots=True)
class _TargetOrbitIntertwiner:
    """Left inverse of one target orbit, split by exact image key."""

    offset: int
    dimension: int
    dual_blocks: dict[InteractionKey, np.ndarray]


def build_wick_to_taylor_transform(calculations, covariance) -> sparse.csr_matrix:
    """Map symmetry-reduced Wick parameters to ordinary Taylor IFC parameters.

    The map is the identity plus same-parity contractions with the displacement
    covariance.  Building it in the irreducible orbit basis lets Taylor ASR and
    rotational constraints be imposed exactly during a Wick-basis fit.
    """
    dimensions = [_parameter_count(calculation) for calculation in calculations]
    offsets = np.cumsum([0, *dimensions])
    total = int(offsets[-1])
    transform = sparse.eye(total, format="lil", dtype=float)
    covariance = np.asarray(covariance).reshape(
        len(calculations[0].supercell), 3, len(calculations[0].supercell), 3
    )
    by_order = {
        calculation.config.order: (index, calculation)
        for index, calculation in enumerate(calculations)
    }
    image_maps = {}
    orbit_intertwiners = {}
    for order, (index, calculation) in by_order.items():
        images = {}
        grouped = {}
        for key, _cluster, columns, local_offset in _primitive_image_columns(calculation):
            global_offset = int(offsets[index] + local_offset)
            images[key] = (global_offset, columns)
            grouped.setdefault(global_offset, []).append((key, columns))
        image_maps[order] = images
        orbit_intertwiners[order] = tuple(
            _target_orbit_intertwiner(offset, target_images)
            for offset, target_images in grouped.items()
        )

    for target_order in by_order:
        for source_order in range(target_order + 2, max(by_order, default=0) + 1, 2):
            if source_order not in by_order:
                continue
            source_index, source = by_order[source_order]
            pairs = (source_order - target_order) // 2
            coefficient = (-1.0) ** pairs / (2.0**pairs * factorial(pairs))
            contracted_by_target: dict[tuple[int, ...], dict[int, np.ndarray]] = {}
            contraction_scales: dict[tuple[int, ...], dict[int, np.ndarray]] = {}
            for key, cluster, columns, local_offset in _primitive_image_columns(source):
                contracted = columns.reshape((3,) * source_order + (-1,))
                for pair in reversed(range(pairs)):
                    left = target_order + 2 * pair
                    contracted = np.einsum(
                        "...abp,ab->...p",
                        contracted,
                        covariance[cluster[left], :, cluster[left + 1], :],
                        optimize=True,
                    )
                target_cluster = InteractionKey.from_labels(key.labels[:target_order])
                source_offset = int(offsets[source_index] + local_offset)
                contributions = contracted_by_target.setdefault(target_cluster, {})
                contribution = contracted.reshape(3**target_order, -1)
                contributions[source_offset] = (
                    contributions.get(source_offset, np.zeros_like(contribution)) + contribution
                )
                scales = contraction_scales.setdefault(target_cluster, {})
                scales[source_offset] = scales.get(
                    source_offset, np.zeros_like(contribution)
                ) + np.abs(contribution)
            _validate_missing_exact_contractions(
                contracted_by_target,
                contraction_scales,
                image_maps[target_order],
                source_order=source_order,
                target_order=target_order,
            )
            # The target-orbit dual is independent of the source order and of
            # the covariance.  Apply its exact-image blocks directly instead
            # of rebuilding the same joint least-squares problem here.
            for intertwiner in orbit_intertwiners[target_order]:
                source_offsets = sorted(
                    {
                        source_offset
                        for key in intertwiner.dual_blocks
                        for source_offset in contracted_by_target.get(key, {})
                    }
                )
                if not source_offsets:
                    continue
                for source_offset in source_offsets:
                    mapping = None
                    for key, dual in intertwiner.dual_blocks.items():
                        contribution = contracted_by_target.get(key, {}).get(source_offset)
                        if contribution is None:
                            continue
                        value = dual @ contribution
                        mapping = value if mapping is None else mapping + value
                    assert mapping is not None
                    width = mapping.shape[1]
                    transform[
                        intertwiner.offset : intertwiner.offset + intertwiner.dimension,
                        source_offset : source_offset + width,
                    ] += coefficient * mapping
    return transform.tocsr()


def _target_orbit_intertwiner(
    offset: int,
    images: list[tuple[InteractionKey, np.ndarray]],
) -> _TargetOrbitIntertwiner:
    """Factor one target orbit once and return its image-wise dual blocks."""
    columns = np.vstack([values for _key, values in images])
    q, r = qr(columns, mode="economic", pivoting=False, check_finite=False)
    dimension = columns.shape[1]
    threshold = (
        np.finfo(float).eps * max(columns.shape) * max(float(np.max(np.abs(np.diag(r)))), 1.0)
    )
    if np.any(np.abs(np.diag(r)) <= threshold):
        raise RuntimeError("target orbit image columns are rank deficient")
    dual = solve_triangular(r, q.T, lower=False, check_finite=False)
    blocks = {}
    begin = 0
    for key, values in images:
        end = begin + values.shape[0]
        blocks[key] = dual[:, begin:end]
        begin = end
    return _TargetOrbitIntertwiner(int(offset), dimension, blocks)


def _validate_missing_exact_contractions(
    contracted_by_target,
    contraction_scales,
    target_images,
    *,
    source_order,
    target_order,
    absolute_tolerance=1e-12,
    relative_tolerance=1e-9,
):
    """Reject exact lower-order interactions omitted by configured support."""
    for key in sorted(set(contracted_by_target).difference(target_images)):
        magnitude = max(
            (float(np.max(np.abs(values))) for values in contracted_by_target[key].values()),
            default=0.0,
        )
        scale = max(
            (float(np.max(values)) for values in contraction_scales.get(key, {}).values()),
            default=0.0,
        )
        threshold = absolute_tolerance + relative_tolerance * scale
        if magnitude <= threshold:
            continue
        raise ValueError(
            f"Wick-to-Taylor FC{source_order}->FC{target_order} contraction creates "
            f"exact interaction {key} outside the configured FC{target_order} support "
            f"(maximum coefficient={magnitude:.6e})"
        )


def lowered_fc1(calculations, parameters, covariance) -> np.ndarray:
    """Return the diagnostic Taylor FC1 generated by odd Wick orders."""
    transform = build_fc1_lowering_transform(calculations, covariance)
    return np.asarray(transform @ np.asarray(parameters)).reshape(-1, 3)


def build_fc1_lowering_transform(calculations, covariance) -> sparse.csr_matrix:
    """Map Wick parameters to Taylor FC1 at the reference structure."""
    covariance = np.asarray(covariance).reshape(
        len(calculations[0].supercell), 3, len(calculations[0].supercell), 3
    )
    n_primitive = calculations[0].index.n_primitive
    total = sum(_parameter_count(calculation) for calculation in calculations)
    matrix_rows = []
    matrix_columns = []
    matrix_data = []
    offset = 0
    for calculation in calculations:
        dimension = _parameter_count(calculation)
        order = calculation.config.order
        if order % 2:
            pairs = (order - 1) // 2
            coefficient = (-1.0) ** pairs / (2.0**pairs * factorial(pairs))
            for key, cluster, columns, local_offset in _primitive_image_columns(calculation):
                contracted = columns.reshape((3,) * order + (-1,))
                for pair in reversed(range(pairs)):
                    left = 1 + 2 * pair
                    contracted = np.einsum(
                        "...abp,ab->...p",
                        contracted,
                        covariance[cluster[left], :, cluster[left + 1], :],
                        optimize=True,
                    )
                for direction in range(3):
                    coefficients = coefficient * contracted[direction]
                    nonzero = np.flatnonzero(np.abs(coefficients) > 1e-12)
                    primitive_site = int(key.sites[0])
                    matrix_rows.extend([primitive_site * 3 + direction] * len(nonzero))
                    matrix_columns.extend(offset + local_offset + int(value) for value in nonzero)
                    matrix_data.extend(float(coefficients[value]) for value in nonzero)
        offset += dimension
    return sparse.coo_matrix(
        (matrix_data, (matrix_rows, matrix_columns)),
        shape=(n_primitive * 3, total),
    ).tocsr()


def _parameter_count(calculation):
    return sum(orbit.dimension for orbit in calculation.orbit_space.orbits)


def _primitive_image_columns(calculation):
    """Yield exact key, finite realization, and parameter columns."""
    space = getattr(calculation, "primitive_orbit_space", None)
    if space is None:
        space = calculation.interaction_space.primitive_orbit_space
    offset = 0
    for orbit in space.orbits:
        representative = np.linalg.solve(orbit.basis[orbit.pivots].T, orbit.basis.T).T
        for image in orbit.images:
            key = image.key
            cluster = (calculation.index.representative(key.sites[0]),) + tuple(
                calculation.index.atom(site, translation)
                for site, translation in zip(key.sites[1:], key.translations, strict=True)
            )
            yield key, cluster, image.action.apply_columns(representative), offset
        offset += orbit.dimension
