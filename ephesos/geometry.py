"""Projected occultor geometry shared by modeling and visualization."""

from typing import Literal

import numpy as np


OccultorType = Literal[
    "disk",
    "oblate",
    "face_on_rings",
    "horizontal_rings",
    "vertical_rings",
    "custom",
]


def _sample_custom_occultor_mask(
    horizontal_offset: np.ndarray,
    vertical_offset: np.ndarray,
    equivalent_radius_ratio: float,
    custom_mask: np.ndarray,
    custom_mask_extent: tuple[float, float, float, float],
) -> np.ndarray:
    """Sample a raster silhouette whose coordinates use the benchmark radius."""

    custom_mask = np.asarray(custom_mask, dtype=bool)
    if custom_mask.ndim != 2 or min(custom_mask.shape) < 2 or not np.any(custom_mask):
        raise ValueError("custom_mask must be a nonempty two-dimensional array")
    if len(custom_mask_extent) != 4 or not np.isfinite(custom_mask_extent).all():
        raise ValueError("custom_mask_extent must contain four finite values")
    left, right, bottom, top = custom_mask_extent
    if left >= right or bottom >= top:
        raise ValueError("custom_mask_extent bounds must be increasing")

    normalized_x = horizontal_offset / equivalent_radius_ratio
    normalized_y = vertical_offset / equivalent_radius_ratio
    column = np.rint((normalized_x - left) * (custom_mask.shape[1] - 1) / (right - left)).astype(int)
    row = np.rint((normalized_y - bottom) * (custom_mask.shape[0] - 1) / (top - bottom)).astype(int)
    inside = (
        (column >= 0)
        & (column < custom_mask.shape[1])
        & (row >= 0)
        & (row < custom_mask.shape[0])
    )
    occulted = np.zeros(np.broadcast_shapes(horizontal_offset.shape, vertical_offset.shape), dtype=bool)
    occulted[inside] = custom_mask[row[inside], column[inside]]
    return occulted


def _inclined_ring_area_factor() -> float:
    """Return the union area of the unit planet and projected ring ellipse."""

    semimajor_axis = 1.75
    semiminor_axis = 0.2
    crossing_x = np.sqrt((1.0 - semiminor_axis**2) / (1.0 - semiminor_axis**2 / semimajor_axis**2))

    def circle_integral(x_coordinate: float) -> float:
        return 0.5 * (x_coordinate * np.sqrt(1.0 - x_coordinate**2) + np.arcsin(x_coordinate))

    def ellipse_integral(x_coordinate: float) -> float:
        normalized_x = x_coordinate / semimajor_axis
        return (
            0.5
            * semiminor_axis
            * (
                x_coordinate * np.sqrt(1.0 - normalized_x**2)
                + semimajor_axis * np.arcsin(normalized_x)
            )
        )

    intersection_area = 4.0 * (
        ellipse_integral(crossing_x) + circle_integral(1.0) - circle_integral(crossing_x)
    )
    return 1.0 + semimajor_axis * semiminor_axis - intersection_area / np.pi


def projected_occultor_mask(
    horizontal_offset: np.ndarray,
    vertical_offset: np.ndarray,
    equivalent_radius_ratio: float,
    occultor_type: OccultorType,
    *,
    oblateness: float = 0.3,
    custom_mask: np.ndarray | None = None,
    custom_mask_extent: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
) -> np.ndarray:
    """Return a mask whose projected area is pi times the equivalent radius squared."""

    if equivalent_radius_ratio <= 0.0:
        raise ValueError("equivalent_radius_ratio must be positive")
    if not 0.0 <= oblateness < 1.0:
        raise ValueError("oblateness must be between zero and one")

    radial_distance = np.sqrt(horizontal_offset**2 + vertical_offset**2)
    if occultor_type == "custom":
        if custom_mask is None:
            raise ValueError("custom_mask is required for a custom occultor")
        return _sample_custom_occultor_mask(
            horizontal_offset,
            vertical_offset,
            equivalent_radius_ratio,
            custom_mask,
            custom_mask_extent,
        )
    if occultor_type == "disk":
        return radial_distance <= equivalent_radius_ratio
    if occultor_type == "oblate":
        axis_ratio = 1.0 - oblateness
        semimajor_axis = equivalent_radius_ratio / np.sqrt(axis_ratio)
        semiminor_axis = equivalent_radius_ratio * np.sqrt(axis_ratio)
        return (horizontal_offset / semimajor_axis) ** 2 + (
            vertical_offset / semiminor_axis
        ) ** 2 <= 1.0

    if occultor_type == "face_on_rings":
        area_factor = 1.0 + 1.75**2 - 1.5**2
    elif occultor_type in ("horizontal_rings", "vertical_rings"):
        area_factor = _inclined_ring_area_factor()
    else:
        raise ValueError(f"unsupported occultor_type: {occultor_type}")

    planet_radius = equivalent_radius_ratio / np.sqrt(area_factor)
    occulted = radial_distance <= planet_radius
    if occultor_type == "face_on_rings":
        return occulted | (
            (radial_distance >= 1.5 * planet_radius) & (radial_distance <= 1.75 * planet_radius)
        )

    if occultor_type == "horizontal_rings":
        ring_mask = (horizontal_offset / (1.75 * planet_radius)) ** 2 + (
            vertical_offset / (0.2 * planet_radius)
        ) ** 2 <= 1.0
    else:
        ring_mask = (horizontal_offset / (0.2 * planet_radius)) ** 2 + (
            vertical_offset / (1.75 * planet_radius)
        ) ** 2 <= 1.0
    return occulted | ring_mask
