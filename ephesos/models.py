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


def evaluate_multiplanet_transit_model(
    time_days: np.ndarray,
    *,
    period_days: np.ndarray,
    transit_epoch_days: np.ndarray,
    radius_ratio: np.ndarray,
    summed_radius_to_semimajor_axis: np.ndarray,
    cosine_inclination: np.ndarray,
    limb_darkening_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the combined light curve of multiple transiting planets."""

    time_days = np.asarray(time_days, dtype=float)
    parameters = tuple(
        np.asarray(parameter, dtype=float)
        for parameter in (
            period_days,
            transit_epoch_days,
            radius_ratio,
            summed_radius_to_semimajor_axis,
            cosine_inclination,
        )
    )
    if time_days.ndim != 1 or time_days.size < 2 or not np.isfinite(time_days).all():
        raise ValueError("time_days must be a finite one-dimensional array")
    if any(parameter.ndim != 1 for parameter in parameters):
        raise ValueError("planet parameters must be one-dimensional arrays")
    if len({parameter.size for parameter in parameters}) != 1 or parameters[0].size < 2:
        raise ValueError("planet parameters must have matching lengths of at least two")
    if not all(np.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("planet parameters must contain only finite values")

    period_days, transit_epoch_days, radius_ratio, summed_radius, cosine_inclination = parameters
    if np.any(period_days <= 0.0):
        raise ValueError("period_days must be positive")
    if np.any(radius_ratio <= 0.0):
        raise ValueError("radius_ratio must be positive")
    if np.any(summed_radius <= 0.0):
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if np.any(np.abs(cosine_inclination) > 1.0):
        raise ValueError("cosine_inclination must be between -1 and 1")

    result = eval_modl(
        time_days,
        "PlanetarySystem",
        pericomp=period_days,
        epocmtracomp=transit_epoch_days,
        rsmacomp=summed_radius,
        cosicomp=cosine_inclination,
        rratcomp=radius_ratio,
        coeflmdk=limb_darkening_coefficients,
        typelmdk="quad",
        booldiag=False,
        typeverb=0,
    )
    return result["rflx"][:, 0]


def mutual_hill_separations(
    period_days: np.ndarray,
    planet_mass_earth: np.ndarray,
    stellar_mass_solar: float,
) -> np.ndarray:
    """Return adjacent orbital separations in mutual Hill radii."""

    period_days = np.asarray(period_days, dtype=float)
    planet_mass_earth = np.asarray(planet_mass_earth, dtype=float)
    if period_days.ndim != 1 or period_days.shape != planet_mass_earth.shape:
        raise ValueError("period_days and planet_mass_earth must be matching arrays")
    if period_days.size < 2 or np.any(np.diff(period_days) <= 0.0):
        raise ValueError("period_days must contain at least two increasing values")
    if np.any(planet_mass_earth <= 0.0) or stellar_mass_solar <= 0.0:
        raise ValueError("planet and stellar masses must be positive")

    semimajor_axis_scale = period_days ** (2.0 / 3.0)
    earth_to_solar_mass = 3.0035e-6
    adjacent_mass_ratio = (
        earth_to_solar_mass
        * (planet_mass_earth[:-1] + planet_mass_earth[1:])
        / (3.0 * stellar_mass_solar)
    ) ** (1.0 / 3.0)
    mutual_hill_radius = (
        0.5 * (semimajor_axis_scale[:-1] + semimajor_axis_scale[1:]) * adjacent_mass_ratio
    )
    return np.diff(semimajor_axis_scale) / mutual_hill_radius


def derive_transit_features(
    time_days: np.ndarray,
    relative_flux: np.ndarray,
) -> np.ndarray:
    """Derive depth, duration, ingress time, and equivalent width from a transit."""

    time_days = np.asarray(time_days, dtype=float)
    relative_flux = np.asarray(relative_flux, dtype=float)
    if time_days.ndim != 1 or relative_flux.shape != time_days.shape or time_days.size < 3:
        raise ValueError("time_days and relative_flux must be matching one-dimensional arrays")
    if not np.isfinite(time_days).all() or not np.isfinite(relative_flux).all():
        raise ValueError("light-curve inputs must contain only finite values")
    if not np.all(np.diff(time_days) > 0.0):
        raise ValueError("time_days must be strictly increasing")

    flux_deficit = np.clip(1.0 - relative_flux, 0.0, None)
    depth = np.max(flux_deficit)
    if depth <= 0.0:
        raise ValueError("relative_flux must contain a transit")

    center_index = int(np.argmax(flux_deficit))
    ingress_deficit = flux_deficit[: center_index + 1]
    ingress_time = time_days[: center_index + 1]
    egress_deficit = flux_deficit[center_index:][::-1]
    egress_time = time_days[center_index:][::-1]
    contact_start = np.interp(0.001 * depth, ingress_deficit, ingress_time)
    contact_end = np.interp(0.001 * depth, egress_deficit, egress_time)
    ingress_start = np.interp(0.01 * depth, ingress_deficit, ingress_time)
    ingress_end = np.interp(0.99 * depth, ingress_deficit, ingress_time)

    depth_percent = 100.0 * depth  # [percent]
    duration_hours = 24.0 * (contact_end - contact_start)  # [hour]
    ingress_minutes = 24.0 * 60.0 * (ingress_end - ingress_start)  # [minute]
    equivalent_width_minutes = 24.0 * 60.0 * np.trapezoid(flux_deficit, time_days)  # [minute]
    return np.array((depth_percent, duration_hours, ingress_minutes, equivalent_width_minutes))


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
