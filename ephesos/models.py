"""Concise public wrappers around the Ephesos forward model."""

import numpy as np
import chalcedon
from astropy.constants import G, M_sun, R_sun, c
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from chalcedon import evaluate_self_lensing_model
from tdpy.exoplanet import (
    quadratic_limb_darkened_stellar_grid,
    retr_deptbeam,
    retr_deptelli,
)

from .geometry import OccultorType, projected_occultor_mask
from .main import eval_modl


def _validate_unmasked_real_inputs(values, description):
    if any(np.ma.is_masked(value) or np.iscomplexobj(value) for value in values):
        raise ValueError(f"{description} must contain unmasked real values")


def evaluate_compact_object_signatures(
    period_days,
    companion_mass_solar,
    stellar_radius_solar=1.0,  # [R_Sun]
    stellar_mass_solar=1.0,  # [M_Sun]
    stellar_density_cgs=1.41,  # [g cm^-3]
) -> dict[str, np.ndarray]:
    """Predict compact-object beaming, ellipsoidal, and self-lensing amplitudes in ppt."""

    inputs = (period_days, companion_mass_solar, stellar_radius_solar,
              stellar_mass_solar, stellar_density_cgs)
    _validate_unmasked_real_inputs(inputs, "Compact-object signature inputs")
    parameters = np.broadcast_arrays(
        *(np.asarray(value, dtype=float) for value in inputs)
    )
    if not all(np.isfinite(value).all() for value in parameters):
        raise ValueError("Compact-object signature inputs must be finite.")
    if any(np.any(value <= 0.0) for value in parameters):
        raise ValueError("Periods, masses, radii, and density must be positive.")

    (
        period_days,
        companion_mass_solar,
        stellar_radius_solar,
        stellar_mass_solar,
        stellar_density_cgs,
    ) = parameters
    return {
        "beaming": np.asarray(
            retr_deptbeam(period_days, stellar_mass_solar, companion_mass_solar)
        ),
        "ellipsoidal": np.asarray(
            retr_deptelli(
                period_days, stellar_density_cgs, stellar_mass_solar, companion_mass_solar
            )
        ),
        "self_lensing": np.asarray(
            chalcedon.retr_amplslen(
                period_days, stellar_radius_solar, companion_mass_solar, stellar_mass_solar
            )
        ),
    }


def evaluate_linear_transit_times(transit_epoch, epoch_time, orbital_period):
    """Predict a linear transit ephemeris in the units of epoch_time and orbital_period."""

    return epoch_time + orbital_period * np.asarray(transit_epoch)


def evaluate_sinusoidal_ttv(transit_epoch, offset, phase, amplitude, ttv_period):
    """Predict sinusoidal transit-timing residuals in the units of offset and amplitude."""

    return offset + amplitude * np.sin(
        phase + 2.0 * np.pi * np.asarray(transit_epoch) / ttv_period
    )


def evaluate_sinusoidal_transit_times(
    transit_epochs: np.ndarray,
    *,
    epoch_time_days: float,
    orbital_period_days: float,
    ttv_period_epochs: float,
    ttv_amplitude_days: float,
    ttv_phase_radians: float = 0.0,
    ttv_offset_days: float = 0.0,
) -> np.ndarray:
    """Predict transit times from a linear ephemeris plus a sinusoidal TTV."""

    transit_epochs = np.asarray(transit_epochs, dtype=float)
    parameters = np.asarray(
        (
            epoch_time_days,
            orbital_period_days,
            ttv_period_epochs,
            ttv_amplitude_days,
            ttv_phase_radians,
            ttv_offset_days,
        ),
        dtype=float,
    )
    if transit_epochs.ndim != 1 or not np.isfinite(transit_epochs).all():
        raise ValueError("transit_epochs must be a finite one-dimensional array")
    if not np.isfinite(parameters).all():
        raise ValueError("transit-time parameters must be finite")
    if orbital_period_days <= 0.0 or ttv_period_epochs <= 0.0:
        raise ValueError("orbital_period_days and ttv_period_epochs must be positive")

    linear_times = evaluate_linear_transit_times(
        transit_epochs, epoch_time_days, orbital_period_days
    )  # [day]
    timing_residuals = evaluate_sinusoidal_ttv(
        transit_epochs,
        ttv_offset_days,
        ttv_phase_radians,
        ttv_amplitude_days,
        ttv_period_epochs,
    )  # [day]
    return linear_times + timing_residuals


def evaluate_nbody_transit_times(
    planet_parameters,
    stellar_mass_solar,
    start_time_days,
    step_days,
    end_time_days,
):
    """Predict transit times with TTVFast through an absolute end time in days.

    Each planet row contains mass in solar masses, period in days, eccentricity,
    inclination, longitude of ascending node, argument of periapsis, and mean
    anomaly. Angular orbital elements use degrees, as required by TTVFast.
    """

    planet_parameters = np.asarray(planet_parameters, dtype=float)
    if (
        planet_parameters.ndim != 2
        or planet_parameters.shape[1] != 7
        or planet_parameters.shape[0] == 0
        or not np.isfinite(planet_parameters).all()
    ):
        raise ValueError("planet_parameters must be a finite array with seven columns")
    if not np.isfinite((stellar_mass_solar, start_time_days, step_days, end_time_days)).all():
        raise ValueError("N-body timing inputs must be finite")
    if stellar_mass_solar <= 0.0 or step_days <= 0.0 or end_time_days <= start_time_days:
        raise ValueError(
            "stellar mass and step must be positive and end_time_days must follow start_time_days"
        )

    import ttvfast

    planets = [ttvfast.models.Planet(*parameters) for parameters in planet_parameters]
    results = ttvfast.ttvfast(
        planets, stellar_mass_solar, start_time_days, step_days, end_time_days
    )
    planet_indices = np.asarray(results["positions"][0])
    transit_epochs = np.asarray(results["positions"][1])
    transit_times = np.asarray(results["positions"][2])
    predictions = []
    for planet_index in range(len(planets)):
        valid = (planet_indices == planet_index) & (transit_times != -2.0)
        order = np.argsort(transit_epochs[valid])
        predictions.append((transit_epochs[valid][order], transit_times[valid][order]))
    return predictions


def _validate_time_days(time_days: np.ndarray, minimum_size: int = 2) -> np.ndarray:
    """Return a finite one-dimensional time array with enough samples."""

    time_days = np.asarray(time_days, dtype=float)
    if time_days.ndim != 1 or time_days.size < minimum_size or not np.isfinite(time_days).all():
        raise ValueError("time_days must be a finite one-dimensional array")
    return time_days


def _validate_grid_size(grid_size: int) -> None:
    """Require an odd grid with enough pixels to resolve the stellar disk."""

    if grid_size < 101 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer of at least 101")


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

    time_days = _validate_time_days(time_days)
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


def evaluate_exomoon_transit_model(
    time_days: np.ndarray,
    *,
    planet_period_days: float,
    planet_radius_ratio: float,
    summed_radius_to_semimajor_axis: float,
    moon_radius_ratio: float,
    moon_separation_planet_radii: float,
    moon_period_days: float,
    moon_orbital_phase_radians: float = 0.0,
    cosine_inclination: float = 0.0,
    limb_darkening_coefficients: tuple[float, float] = (0.4, 0.25),
    grid_size: int = 301,
) -> np.ndarray:
    """Evaluate a circular, coplanar planet-moon transit on a stellar image grid."""

    time_days = _validate_time_days(time_days)  # [day]
    _validate_grid_size(grid_size)
    if not np.isfinite((
        planet_period_days,
        planet_radius_ratio,
        summed_radius_to_semimajor_axis,
        moon_radius_ratio,
        moon_separation_planet_radii,
        moon_period_days,
        moon_orbital_phase_radians,
        cosine_inclination,
        *limb_darkening_coefficients,
    )).all():
        raise ValueError("exomoon model parameters must be finite")
    if planet_period_days <= 0.0 or moon_period_days <= 0.0:
        raise ValueError("planet_period_days and moon_period_days must be positive")
    if planet_radius_ratio <= 0.0:
        raise ValueError("planet_radius_ratio must be positive")
    if moon_radius_ratio < 0.0:
        raise ValueError("moon_radius_ratio must be nonnegative")
    if moon_separation_planet_radii <= 0.0:
        raise ValueError("moon_separation_planet_radii must be positive")
    if summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if not -1.0 <= cosine_inclination <= 1.0:
        raise ValueError("cosine_inclination must be between -1 and 1")
    if not np.isfinite(moon_orbital_phase_radians):
        raise ValueError("moon_orbital_phase_radians must be finite")

    image_x, image_y, _, stellar_brightness = quadratic_limb_darkened_stellar_grid(
        grid_size, limb_darkening_coefficients
    )
    unocculted_flux = stellar_brightness.sum()
    planet_semimajor_axis_stellar_radii = (
        1.0 + planet_radius_ratio
    ) / summed_radius_to_semimajor_axis  # [R_star]
    moon_separation_stellar_radii = (
        moon_separation_planet_radii * planet_radius_ratio
    )  # [R_star]
    planet_phase = 2.0 * np.pi * time_days / planet_period_days  # [rad]
    moon_phase = (
        2.0 * np.pi * time_days / moon_period_days + moon_orbital_phase_radians
    )  # [rad]
    inclination_sine = np.sqrt(1.0 - cosine_inclination**2)
    planet_x = planet_semimajor_axis_stellar_radii * np.sin(planet_phase)  # [R_star]
    planet_y = (
        planet_semimajor_axis_stellar_radii * cosine_inclination * np.cos(planet_phase)
    )  # [R_star]
    planet_z = (
        planet_semimajor_axis_stellar_radii * inclination_sine * np.cos(planet_phase)
    )  # [R_star]
    moon_x = moon_separation_stellar_radii * np.sin(moon_phase)  # [R_star]
    moon_y = moon_separation_stellar_radii * cosine_inclination * np.cos(moon_phase)  # [R_star]
    moon_z = moon_separation_stellar_radii * inclination_sine * np.cos(moon_phase)  # [R_star]

    relative_flux = np.ones(time_days.size)
    for time_index in range(time_days.size):
        occulted = np.zeros(stellar_brightness.shape, dtype=bool)
        if planet_z[time_index] > 0.0:
            occulted |= (
                (image_x - planet_x[time_index]) ** 2
                + (image_y - planet_y[time_index]) ** 2
                <= planet_radius_ratio**2
            )
        if moon_radius_ratio > 0.0 and planet_z[time_index] + moon_z[time_index] > 0.0:
            moon_center_x = planet_x[time_index] + moon_x[time_index]  # [R_star]
            moon_center_y = planet_y[time_index] + moon_y[time_index]  # [R_star]
            occulted |= (
                (image_x - moon_center_x) ** 2 + (image_y - moon_center_y) ** 2
                <= moon_radius_ratio**2
            )
        relative_flux[time_index] -= stellar_brightness[occulted].sum() / unocculted_flux
    return relative_flux


def evaluate_emitting_companion_model(
    time_days: np.ndarray,
    *,
    period_days: float,
    radius_ratio: float,
    summed_radius_to_semimajor_axis: float,
    companion_brightness_ratio: float,
    cosine_inclination: float = 0.0,
    limb_darkening_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate transit, planetary emission, and secondary eclipse."""

    time_days = _validate_time_days(time_days)
    if period_days <= 0.0:
        raise ValueError("period_days must be positive")
    if radius_ratio <= 0.0:
        raise ValueError("radius_ratio must be positive")
    if summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if companion_brightness_ratio < 0.0:
        raise ValueError("companion_brightness_ratio must be nonnegative")
    if not -1.0 <= cosine_inclination <= 1.0:
        raise ValueError("cosine_inclination must be between -1 and 1")

    result = eval_modl(
        time_days,
        "PlanetarySystemEmittingCompanion",
        pericomp=np.array([period_days]),  # [day]
        epocmtracomp=np.array([0.0]),  # [day]
        rsmacomp=np.array([summed_radius_to_semimajor_axis]),
        cosicomp=np.array([cosine_inclination]),
        rratcomp=np.array([radius_ratio]),
        typebrgtcomp="isot",
        ratibrgtcomp=np.array([companion_brightness_ratio]),
        coeflmdk=limb_darkening_coefficients,
        typelmdk="quad",
        booldiag=False,
        typeverb=0,
    )
    return result["rflx"][:, 0]


def evaluate_eclipse_mapping_model(
    time_days: np.ndarray,
    *,
    period_days: float,
    radius_ratio: float,
    summed_radius_to_semimajor_axis: float,
    companion_brightness_map: np.ndarray,
    transit_epoch_days: float = 0.0,
    cosine_inclination: float = 0.0,
    limb_darkening_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate a transit and secondary eclipse for a projected companion map.

    The square brightness map spans -1 to 1 in companion-radius units along
    both sky-plane axes. Pixels outside the unit disk are ignored, and map
    values are relative to the star's disk-averaged surface brightness. The
    projected map remains fixed during the orbit.
    """

    time_days = _validate_time_days(time_days)
    parameters = np.asarray(
        (
            period_days,
            radius_ratio,
            summed_radius_to_semimajor_axis,
            transit_epoch_days,
            cosine_inclination,
        ),
        dtype=float,
    )
    if not np.isfinite(parameters).all():
        raise ValueError("eclipse-mapping parameters must be finite")
    if period_days <= 0.0:
        raise ValueError("period_days must be positive")
    if radius_ratio <= 0.0:
        raise ValueError("radius_ratio must be positive")
    if summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if not -1.0 <= cosine_inclination <= 1.0:
        raise ValueError("cosine_inclination must be between -1 and 1")

    companion_brightness_map = np.asarray(companion_brightness_map, dtype=float)
    if (
        companion_brightness_map.ndim != 2
        or companion_brightness_map.shape[0] != companion_brightness_map.shape[1]
        or companion_brightness_map.shape[0] < 11
    ):
        raise ValueError("companion_brightness_map must be a square raster of at least 11 by 11")
    if not np.isfinite(companion_brightness_map).all() or np.any(companion_brightness_map < 0.0):
        raise ValueError("companion_brightness_map must contain finite nonnegative values")

    map_coordinates = np.linspace(-1.0, 1.0, companion_brightness_map.shape[0])
    map_x, map_y = np.meshgrid(map_coordinates, map_coordinates)
    companion_disk = map_x**2 + map_y**2 <= 1.0
    disk_pixel_count = np.count_nonzero(companion_disk)
    brightness_sum = np.sum(companion_brightness_map[companion_disk])
    companion_flux = np.full(
        time_days.size,
        radius_ratio**2 * brightness_sum / disk_pixel_count,
    )

    orbital_phase = 2.0 * np.pi * (time_days - transit_epoch_days) / period_days
    semimajor_axis_stellar_radii = (1.0 + radius_ratio) / summed_radius_to_semimajor_axis
    companion_x = semimajor_axis_stellar_radii * np.sin(orbital_phase)
    companion_y = semimajor_axis_stellar_radii * cosine_inclination * np.cos(orbital_phase)
    companion_z = (
        semimajor_axis_stellar_radii * np.sqrt(1.0 - cosine_inclination**2) * np.cos(orbital_phase)
    )

    for time_index in np.flatnonzero(companion_z < 0.0):
        projected_x = companion_x[time_index] + radius_ratio * map_x
        projected_y = companion_y[time_index] + radius_ratio * map_y
        companion_occulted = companion_disk & (projected_x**2 + projected_y**2 <= 1.0)
        visible_brightness = np.sum(companion_brightness_map[companion_disk & ~companion_occulted])
        companion_flux[time_index] = radius_ratio**2 * visible_brightness / disk_pixel_count

    stellar_flux = evaluate_transit_model(
        time_days - transit_epoch_days,
        period_days=period_days,
        radius_ratio=radius_ratio,
        summed_radius_to_semimajor_axis=summed_radius_to_semimajor_axis,
        cosine_inclination=cosine_inclination,
        limb_darkening_coefficients=limb_darkening_coefficients,
    )
    return stellar_flux + companion_flux


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

    time_days = _validate_time_days(time_days)
    planet_parameters = (
        period_days,
        transit_epoch_days,
        radius_ratio,
        summed_radius_to_semimajor_axis,
        cosine_inclination,
    )
    _validate_unmasked_real_inputs(planet_parameters, "planet parameters")
    parameters = tuple(
        np.asarray(parameter, dtype=float)
        for parameter in planet_parameters
    )
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


def derive_transit_feature_departures(
    time_days: np.ndarray,
    relative_flux: np.ndarray,
    benchmark_relative_flux: np.ndarray,
) -> np.ndarray:
    """Return transit features minus those of a matched benchmark model."""

    return derive_transit_features(time_days, relative_flux) - derive_transit_features(
        time_days, benchmark_relative_flux
    )


def derive_transit_timing_variations(
    time_days: np.ndarray,
    relative_flux: np.ndarray,
    transit_epochs: np.ndarray,
    *,
    epoch_time_days: float,
    orbital_period_days: float,
    window_half_width_days: float,
) -> dict[str, np.ndarray]:
    """Estimate event times, TTVs, durations, and TDVs from a normalized light curve.

    The event midpoint is the midpoint between its interpolated 0.1%-depth
    contacts. Incomplete or undetected events are returned as NaN.
    """

    for name, values in (
        ("time_days", time_days),
        ("relative_flux", relative_flux),
        ("transit_epochs", transit_epochs),
        ("epoch_time_days", epoch_time_days),
        ("orbital_period_days", orbital_period_days),
        ("window_half_width_days", window_half_width_days),
    ):
        if np.ma.is_masked(values):
            raise ValueError(f"{name} must not contain masked values")
    time_days = np.asarray(time_days, dtype=float)
    relative_flux = np.asarray(relative_flux, dtype=float)
    transit_epochs = np.asarray(transit_epochs, dtype=float)
    scalar_parameters = np.asarray(
        (epoch_time_days, orbital_period_days, window_half_width_days), dtype=float
    )
    if time_days.ndim != 1 or relative_flux.shape != time_days.shape or time_days.size < 3:
        raise ValueError("time_days and relative_flux must be matching one-dimensional arrays")
    if transit_epochs.ndim != 1 or transit_epochs.size == 0 or not np.isfinite(transit_epochs).all():
        raise ValueError("transit_epochs must be a nonempty finite one-dimensional array")
    if not np.isfinite(time_days).all() or not np.isfinite(relative_flux).all():
        raise ValueError("light-curve inputs must contain only finite values")
    if not np.all(np.diff(time_days) > 0.0):
        raise ValueError("time_days must be strictly increasing")
    if not np.isfinite(scalar_parameters).all():
        raise ValueError("ephemeris and window parameters must be finite")
    if orbital_period_days <= 0.0 or window_half_width_days <= 0.0:
        raise ValueError("orbital period and window half-width must be positive")
    if 2.0 * window_half_width_days >= orbital_period_days:
        raise ValueError("transit windows must not overlap")

    predicted_times = evaluate_linear_transit_times(
        transit_epochs, epoch_time_days, orbital_period_days
    )  # [day]
    measured_times = np.full(transit_epochs.shape, np.nan)  # [day]
    durations = np.full(transit_epochs.shape, np.nan)  # [day]

    for index, predicted_time in enumerate(predicted_times):
        in_window = np.abs(time_days - predicted_time) <= window_half_width_days
        event_time = time_days[in_window]
        event_flux = relative_flux[in_window]
        if event_time.size < 3:
            continue

        deficit = np.clip(1.0 - event_flux, 0.0, None)
        depth = float(np.max(deficit))
        if depth <= 0.0:
            continue
        threshold = 0.001 * depth
        in_transit = deficit > threshold
        boundaries = np.diff(in_transit.astype(int), prepend=0, append=0)
        starts = np.flatnonzero(boundaries == 1)
        stops = np.flatnonzero(boundaries == -1) - 1
        if starts.size != 1 or stops.size != 1:
            continue
        start = int(starts[0])
        stop = int(stops[0])
        if start == 0 or stop == event_time.size - 1:
            continue

        left_fraction = (threshold - deficit[start - 1]) / (deficit[start] - deficit[start - 1])
        right_fraction = (threshold - deficit[stop + 1]) / (deficit[stop] - deficit[stop + 1])
        contact_start = event_time[start - 1] + left_fraction * (
            event_time[start] - event_time[start - 1]
        )  # [day]
        contact_end = event_time[stop + 1] + right_fraction * (
            event_time[stop] - event_time[stop + 1]
        )  # [day]
        measured_times[index] = 0.5 * (contact_start + contact_end)  # [day]
        durations[index] = contact_end - contact_start  # [day]

    valid_durations = durations[np.isfinite(durations)]  # [day]
    duration_reference = np.median(valid_durations) if valid_durations.size else np.nan  # [day]
    return {
        "transit_epochs": transit_epochs,
        "predicted_times_days": predicted_times,
        "measured_times_days": measured_times,
        "timing_variations_days": measured_times - predicted_times,
        "durations_days": durations,
        "duration_variations_days": durations - duration_reference,
    }


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
    custom_occultor_mask: np.ndarray | None = None,
    custom_occultor_extent: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
) -> np.ndarray:
    """Integrate a limb-darkened transit only while the occultor is in front."""

    time_days = _validate_time_days(time_days)
    if period_days <= 0.0:
        raise ValueError("period_days must be positive")
    if summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if not -1.0 <= cosine_inclination <= 1.0:
        raise ValueError("cosine_inclination must be between -1 and 1")
    _validate_grid_size(grid_size)
    image_x, image_y, _, stellar_brightness = quadratic_limb_darkened_stellar_grid(
        grid_size, limb_darkening_coefficients
    )
    unocculted_flux = stellar_brightness.sum()

    semimajor_axis_stellar_radii = (1.0 + equivalent_radius_ratio) / summed_radius_to_semimajor_axis
    orbital_phase = 2.0 * np.pi * time_days / period_days
    projected_x = semimajor_axis_stellar_radii * np.sin(orbital_phase)
    projected_y = semimajor_axis_stellar_radii * cosine_inclination * np.cos(orbital_phase)
    line_of_sight = semimajor_axis_stellar_radii * np.sqrt(1.0 - cosine_inclination**2) * np.cos(orbital_phase)

    # Correlate once, then interpolate the blocked flux at subpixel positions.
    # This avoids staircase artifacts from re-rasterizing a hard mask at each time.
    occultor_kernel = projected_occultor_mask(
        image_x,
        image_y,
        equivalent_radius_ratio,
        occultor_type,
        oblateness=oblateness,
        custom_mask=custom_occultor_mask,
        custom_mask_extent=custom_occultor_extent,
    ).astype(float)
    if occultor_type != "custom":
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
    blocked_flux = np.where(line_of_sight > 0.0, blocked_flux, 0.0)
    return 1.0 - blocked_flux / unocculted_flux


def evaluate_exocomet_transit_model(
    time_days: np.ndarray,
    *,
    period_days: float,
    transit_epoch_days: float,
    nucleus_radius_ratio: float,
    summed_radius_to_semimajor_axis: float,
    tail_length_nucleus_radii: float = 12.0,
    tail_width_nucleus_radii: float = 0.25,
    tail_optical_depth: float = 1.0,
    tail_decay_length_nucleus_radii: float = 5.0,
    tail_widening_per_nucleus_radius: float = 0.02,
    cosine_inclination: float = 0.0,
    limb_darkening_coefficients: tuple[float, float] = (0.4, 0.25),
    grid_size: int = 301,
    mask_grid_size: int = 401,
) -> np.ndarray:
    """Model a repeating exocomet transit with a partially transparent trailing dust tail."""

    values = np.asarray((
        period_days,
        transit_epoch_days,
        nucleus_radius_ratio,
        summed_radius_to_semimajor_axis,
        tail_length_nucleus_radii,
        tail_width_nucleus_radii,
        tail_optical_depth,
        tail_decay_length_nucleus_radii,
        tail_widening_per_nucleus_radius,
    ), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Exocomet model parameters must be finite.")
    if period_days <= 0.0 or nucleus_radius_ratio <= 0.0 or summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("Period, nucleus radius, and summed radius-to-axis ratio must be positive.")
    if (
        tail_length_nucleus_radii <= 0.0
        or tail_width_nucleus_radii <= 0.0
        or tail_optical_depth < 0.0
        or tail_decay_length_nucleus_radii <= 0.0
        or tail_widening_per_nucleus_radius < 0.0
    ):
        raise ValueError("Dust-tail length, width, and decay scale must be positive; opacity parameters nonnegative.")
    if mask_grid_size < 101 or mask_grid_size % 2 == 0:
        raise ValueError("mask_grid_size must be an odd integer of at least 101.")

    maximum_tail_width = tail_width_nucleus_radii + (
        tail_widening_per_nucleus_radius * tail_length_nucleus_radii
    )
    left_extent = -1.0 - tail_length_nucleus_radii
    mask_x = np.linspace(left_extent, 1.0, mask_grid_size)
    mask_y = np.linspace(-maximum_tail_width, maximum_tail_width, mask_grid_size)
    coordinate_x, coordinate_y = np.meshgrid(mask_x, mask_y)
    tail_distance = np.maximum(-coordinate_x - 1.0, 0.0)
    local_width = tail_width_nucleus_radii + tail_widening_per_nucleus_radius * tail_distance
    optical_depth = tail_optical_depth * np.exp(
        -tail_distance / tail_decay_length_nucleus_radii
        -0.5 * (coordinate_y / local_width) ** 2
    )
    dust_opacity = np.where(coordinate_x < -1.0, -np.expm1(-optical_depth), 0.0)
    nucleus_opacity = (coordinate_x**2 + coordinate_y**2 <= 1.0).astype(float)
    opacity_map = np.maximum(dust_opacity, nucleus_opacity)

    return evaluate_projected_occultor_model(
        _validate_time_days(time_days) - transit_epoch_days,
        period_days=period_days,
        equivalent_radius_ratio=nucleus_radius_ratio,
        summed_radius_to_semimajor_axis=summed_radius_to_semimajor_axis,
        occultor_type="custom",
        cosine_inclination=cosine_inclination,
        limb_darkening_coefficients=limb_darkening_coefficients,
        grid_size=grid_size,
        custom_occultor_mask=opacity_map,
        custom_occultor_extent=(
            left_extent,
            1.0,
            -maximum_tail_width,
            maximum_tail_width,
        ),
    )


def evaluate_planet_with_disk_model(
    time_days: np.ndarray,
    *,
    period_days: float,
    planet_radius_ratio: float,
    summed_radius_to_semimajor_axis: float,
    disk_inner_radius_planet_radii: float,
    disk_outer_radius_planet_radii: float,
    disk_inclination_degrees: float,
    disk_optical_depth: float,
    cosine_inclination: float = 0.0,
    disk_position_angle_degrees: float = 0.0,
    limb_darkening_coefficients: tuple[float, float] = (0.4, 0.25),
    grid_size: int = 301,
) -> np.ndarray:
    """Evaluate a transit across an opaque planet and a uniform thin disk annulus."""

    time_days = _validate_time_days(time_days)  # [day]
    _validate_grid_size(grid_size)
    if not np.isfinite((
        period_days,
        planet_radius_ratio,
        summed_radius_to_semimajor_axis,
        disk_inner_radius_planet_radii,
        disk_outer_radius_planet_radii,
        disk_inclination_degrees,
        disk_optical_depth,
        cosine_inclination,
        disk_position_angle_degrees,
        *limb_darkening_coefficients,
    )).all():
        raise ValueError("planet-disk model parameters must be finite")
    if period_days <= 0.0 or planet_radius_ratio <= 0.0:
        raise ValueError("period_days and planet_radius_ratio must be positive")
    if summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if not 1.0 <= disk_inner_radius_planet_radii < disk_outer_radius_planet_radii:
        raise ValueError("disk radii must satisfy 1 <= inner radius < outer radius")
    if not 0.0 <= disk_inclination_degrees <= 90.0:
        raise ValueError("disk_inclination_degrees must be between 0 and 90")
    if disk_optical_depth < 0.0 or not np.isfinite(disk_optical_depth):
        raise ValueError("disk_optical_depth must be finite and nonnegative")
    if not np.isfinite(disk_position_angle_degrees):
        raise ValueError("disk_position_angle_degrees must be finite")
    if not -1.0 <= cosine_inclination <= 1.0:
        raise ValueError("cosine_inclination must be between -1 and 1")

    image_x, image_y, _, stellar_brightness = quadratic_limb_darkened_stellar_grid(
        grid_size, limb_darkening_coefficients
    )
    unocculted_flux = stellar_brightness.sum()
    disk_inclination_radians = np.deg2rad(disk_inclination_degrees)  # [rad]
    disk_cosine_inclination = np.cos(disk_inclination_radians)
    position_angle_radians = np.deg2rad(disk_position_angle_degrees)  # [rad]
    rotated_x = (
        image_x * np.cos(position_angle_radians)
        + image_y * np.sin(position_angle_radians)
    )  # [R_star]
    rotated_y = (
        -image_x * np.sin(position_angle_radians)
        + image_y * np.cos(position_angle_radians)
    )  # [R_star]
    disk_radius_planet_radii = np.sqrt(
        rotated_x**2
        + (rotated_y / max(disk_cosine_inclination, 1e-12)) ** 2
    ) / planet_radius_ratio
    ring_mask = (
        (disk_radius_planet_radii >= disk_inner_radius_planet_radii)
        & (disk_radius_planet_radii <= disk_outer_radius_planet_radii)
    )
    planet_mask = image_x**2 + image_y**2 <= planet_radius_ratio**2
    line_of_sight_optical_depth = disk_optical_depth / max(
        disk_cosine_inclination, 1e-12
    )
    occultor_transmission = np.ones(stellar_brightness.shape)
    occultor_transmission[planet_mask] = 0.0
    occultor_transmission[ring_mask & ~planet_mask] = np.exp(
        -line_of_sight_optical_depth
    )
    blocked_flux_kernel = 1.0 - occultor_transmission
    blocked_flux_grid = fftconvolve(
        stellar_brightness,
        0.5 * (blocked_flux_kernel + blocked_flux_kernel[::-1, ::-1]),
        mode="full",
    )

    planet_semimajor_axis_stellar_radii = (
        1.0 + planet_radius_ratio
    ) / summed_radius_to_semimajor_axis  # [R_star]
    orbital_phase = 2.0 * np.pi * time_days / period_days  # [rad]
    projected_x = planet_semimajor_axis_stellar_radii * np.sin(orbital_phase)  # [R_star]
    projected_y = (
        planet_semimajor_axis_stellar_radii * cosine_inclination * np.cos(orbital_phase)
    )  # [R_star]
    line_of_sight = (
        planet_semimajor_axis_stellar_radii
        * np.sqrt(1.0 - cosine_inclination**2)
        * np.cos(orbital_phase)
    )  # [R_star]
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
    blocked_flux = np.where(line_of_sight > 0.0, blocked_flux, 0.0)
    return 1.0 - blocked_flux / unocculted_flux
