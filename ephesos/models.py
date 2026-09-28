"""Concise public wrappers around the Ephesos forward model."""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

from .geometry import OccultorType, projected_occultor_mask
from .main import eval_modl


def evaluate_transit_model(
    time_days: np.ndarray,
    *,
    period_days: float,
    radius_ratio: float,
    summed_radius_to_semimajor_axis: float,
    cosine_inclination: float = 0.0,
    system_type: str = "PlanetarySystem",
    limb_darkening_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate one deterministic transit and return its relative flux."""

    time_days = np.asarray(time_days, dtype=float)
    if time_days.ndim != 1 or time_days.size < 2 or not np.isfinite(time_days).all():
        raise ValueError("time_days must be a finite one-dimensional array")
    if period_days <= 0.0:
        raise ValueError("period_days must be positive")
    if radius_ratio <= 0.0:
        raise ValueError("radius_ratio must be positive")
    if summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if not -1.0 <= cosine_inclination <= 1.0:
        raise ValueError("cosine_inclination must be between -1 and 1")

    result = eval_modl(
        time_days,
        system_type,
        pericomp=np.array([period_days]),  # [day]
        epocmtracomp=np.array([0.0]),  # [day]
        rsmacomp=np.array([summed_radius_to_semimajor_axis]),
        cosicomp=np.array([cosine_inclination]),
        rratcomp=np.array([radius_ratio]),
        coeflmdk=limb_darkening_coefficients,
        typelmdk="quad",
        booldiag=False,
        typeverb=0,
    )
    relative_flux = result["rflx"][:, 0]
    return relative_flux


def evaluate_projected_occultor_model(
    time_days: np.ndarray,
    *,
    period_days: float,
    equivalent_radius_ratio: float,
    summed_radius_to_semimajor_axis: float,
    occultor_type: OccultorType,
    cosine_inclination: float = 0.0,
    oblateness: float = 0.3,
    limb_darkening_coefficients: tuple[float, float] = (0.4, 0.25),
    grid_size: int = 401,
) -> np.ndarray:
    """Integrate a limb-darkened transit for an area-normalized projected shape."""

    time_days = np.asarray(time_days, dtype=float)
    if time_days.ndim != 1 or time_days.size < 2 or not np.isfinite(time_days).all():
        raise ValueError("time_days must be a finite one-dimensional array")
    if period_days <= 0.0:
        raise ValueError("period_days must be positive")
    if summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if not -1.0 <= cosine_inclination <= 1.0:
        raise ValueError("cosine_inclination must be between -1 and 1")
    if grid_size < 101 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer of at least 101")

    coordinates = np.linspace(-1.0, 1.0, grid_size)
    image_x, image_y = np.meshgrid(coordinates, coordinates)
    radial_distance = np.sqrt(image_x**2 + image_y**2)
    stellar_disk = radial_distance <= 1.0
    cosine_emission_angle = np.sqrt(np.clip(1.0 - radial_distance**2, 0.0, 1.0))
    linear_coefficient, quadratic_coefficient = limb_darkening_coefficients
    stellar_brightness = np.zeros_like(radial_distance)
    stellar_brightness[stellar_disk] = (
        1.0
        - linear_coefficient * (1.0 - cosine_emission_angle[stellar_disk])
        - quadratic_coefficient * (1.0 - cosine_emission_angle[stellar_disk]) ** 2
    )
    unocculted_flux = stellar_brightness.sum()

    semimajor_axis_stellar_radii = (1.0 + equivalent_radius_ratio) / summed_radius_to_semimajor_axis
    orbital_phase = 2.0 * np.pi * time_days / period_days
    projected_x = semimajor_axis_stellar_radii * np.sin(orbital_phase)
    projected_y = semimajor_axis_stellar_radii * cosine_inclination * np.cos(orbital_phase)

    # Correlate once, then interpolate the blocked flux at subpixel positions.
    # This avoids staircase artifacts from re-rasterizing a hard mask at each time.
    occultor_kernel = projected_occultor_mask(
        image_x,
        image_y,
        equivalent_radius_ratio,
        occultor_type,
        oblateness=oblateness,
    ).astype(float)
    occultor_kernel = 0.5 * (occultor_kernel + occultor_kernel[::-1, ::-1])
    blocked_flux_grid = fftconvolve(
        stellar_brightness,
        occultor_kernel,
        mode="full",
    )
    pixels_per_stellar_radius = 0.5 * (grid_size - 1)
    sample_coordinates = np.vstack(
        (
            projected_y * pixels_per_stellar_radius + grid_size - 1,
            projected_x * pixels_per_stellar_radius + grid_size - 1,
        )
    )
    blocked_flux = map_coordinates(
        blocked_flux_grid,
        sample_coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return 1.0 - blocked_flux / unocculted_flux
