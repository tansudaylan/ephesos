"""Photometric forward models for rotating spotted stars and flares."""

import numpy as np
from functools import lru_cache
from tdpy.exoplanet import quadratic_limb_darkening, retr_lcurmodl_flarsing


def retr_protlati(prot, shea, lati):
    """Return periods in prot's units for dimensionless shear and latitudes in degrees."""

    return prot * (1.0 + shea * np.sin(np.deg2rad(lati)) ** 2)


def _aligned_vectors(*values):
    vectors = tuple(np.asarray(value, dtype=float) for value in values)
    if any(vector.ndim != 1 or not np.isfinite(vector).all() for vector in vectors):
        raise ValueError("Model parameters must be finite one-dimensional arrays")
    if len({vector.size for vector in vectors}) != 1:
        raise ValueError("Each spot or flare must have one value for every parameter")
    return vectors


@lru_cache(maxsize=8)
def _spot_quadrature(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angle = 0.25 * np.pi * (nodes + 1.0)  # [rad]
    return np.sin(angle)**2, weights * 0.25 * np.pi * np.sin(2.0 * angle)


def _spot_light_curve(time_days, period_days, longitude, latitude, radius, inclination,
                     contrast, limb_darkening, quadrature_order=64):
    """Integrate a rotating circular spherical cap over the visible stellar disk."""

    latitude = np.deg2rad(latitude)  # [rad]
    inclination = np.deg2rad(inclination)  # [rad]
    phase = np.deg2rad(longitude) + 2.0 * np.pi * np.remainder(time_days / period_days, 1.0)  # [rad]
    center_mu = np.clip(np.sin(inclination) * np.cos(latitude) * np.cos(phase)
                        + np.cos(inclination) * np.sin(latitude), -1.0, 1.0)
    separation = np.arccos(center_mu)  # [rad]
    angular_radius = np.arcsin(radius)  # [rad]
    cosine_radius = np.sqrt(max(0.0, 1.0 - radius**2))
    visible = separation < 0.5 * np.pi + angular_radius
    lower_mu = np.maximum(0.0, np.cos(np.minimum(0.5 * np.pi, separation + angular_radius)))
    upper_mu = np.cos(np.abs(separation - angular_radius))
    upper_mu = np.where(visible, np.maximum(lower_mu, upper_mu), lower_mu)

    linear, quadratic = limb_darkening
    normalization = np.pi * (1.0 - linear / 3.0 - quadratic / 6.0)

    def integrated_intensity(mu):
        return ((1.0 - linear - quadratic) * mu**2 / 2.0
                + (linear + 2.0 * quadratic) * mu**3 / 3.0 - quadratic * mu**4 / 4.0)

    covered_flux = np.where(
        separation < angular_radius,
        2.0 * np.pi * (integrated_intensity(1.0) - integrated_intensity(upper_mu)),
        0.0,
    )
    partial = upper_mu > lower_mu
    if np.any(partial):
        nodes, weights = _spot_quadrature(quadrature_order)
        interval = upper_mu[partial] - lower_mu[partial]
        mu = lower_mu[partial, None] + interval[:, None] * nodes
        denominator = np.sqrt(np.maximum(0.0, 1.0 - mu**2)) * np.sqrt(
            np.maximum(0.0, 1.0 - center_mu[partial, None]**2),
        )
        ring_boundary = np.divide(
            cosine_radius - mu * center_mu[partial, None], denominator,
            out=np.ones_like(mu), where=denominator > 0.0,
        )
        azimuth_width = 2.0 * np.arccos(np.clip(ring_boundary, -1.0, 1.0))  # [rad]
        intensity = quadratic_limb_darkening(mu, limb_darkening)
        covered_flux[partial] += interval * np.sum(azimuth_width * intensity * mu * weights, axis=1)
    return 1.0 - (1.0 - contrast) * covered_flux / normalization


def evaluate_spot_flare_model(
    time_days,
    *,
    rotation_period_days,
    inclination_degrees,
    spot_latitudes_degrees=(),
    spot_longitudes_degrees=(),
    spot_radius_ratios=(),
    limb_darkening=(0.3, 0.3),
    relative_shear=0.0,
    spot_contrast=0.7,
    flux_offset=0.0,
    spot_peak_times_days=None,
    spot_widths_days=None,
    flare_peak_times_days=(),
    flare_amplitudes=(),
    flare_rise_times_days=(),
    flare_decay_times_days=(),
    return_components=False,
    quadrature_order=64,
):
    """Predict additive spot deficits, optional Gaussian spot evolution, and flares.

    Native disk integration treats spots as spherical caps with angular radius
    arcsin(radius ratio), including their partial visibility at the stellar limb.
    The independent-deficit sum does not account for overlapping spots. Flare
    amplitudes are maxima on the supplied time grid, following TDpy's kernel.
    """

    time_days = np.asarray(time_days, dtype=float)
    if time_days.ndim != 1 or time_days.size == 0 or not np.isfinite(time_days).all():
        raise ValueError("time_days must be a nonempty finite one-dimensional array")
    scalar_parameters = np.asarray(
        [rotation_period_days, inclination_degrees, relative_shear, spot_contrast, flux_offset],
        dtype=float,
    )
    if scalar_parameters.shape != (5,) or not np.isfinite(scalar_parameters).all():
        raise ValueError("Stellar parameters must be finite scalars")
    if rotation_period_days <= 0 or not 0 <= spot_contrast <= 1:
        raise ValueError("Rotation period must be positive and spot contrast must lie in [0, 1]")
    limb_darkening = np.asarray(limb_darkening, dtype=float)
    if limb_darkening.shape != (2,) or not np.isfinite(limb_darkening).all():
        raise ValueError("Provide two finite quadratic limb-darkening coefficients")
    latitudes, longitudes, radii = _aligned_vectors(
        spot_latitudes_degrees, spot_longitudes_degrees, spot_radius_ratios,
    )
    extrema = [0.0, 1.0]
    if limb_darkening[1] != 0.0:
        stationary_mu = (limb_darkening[0] + 2.0 * limb_darkening[1]) / (2.0 * limb_darkening[1])
        if 0.0 < stationary_mu < 1.0:
            extrema.append(stationary_mu)
    if latitudes.size and np.any(quadratic_limb_darkening(np.asarray(extrema), limb_darkening) < 0.0):
        raise ValueError("Limb-darkened intensity must be nonnegative across the stellar disk")
    if (isinstance(quadrature_order, (bool, np.bool_))
            or not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 8):
        raise ValueError("quadrature_order must be an integer of at least 8")
    if np.any(np.abs(latitudes) > 90) or np.any((radii <= 0) | (radii > 1)):
        raise ValueError("Spot latitudes must lie in [-90, 90] and radii in (0, 1]")
    periods = retr_protlati(rotation_period_days, relative_shear, latitudes)
    if np.any(periods <= 0):
        raise ValueError("Shear must yield positive spot rotation periods")
    evolution = np.ones((latitudes.size, time_days.size))
    if (spot_peak_times_days is None) != (spot_widths_days is None):
        raise ValueError("Provide both spot peak times and evolution widths")
    if spot_peak_times_days is not None:
        peaks, widths = _aligned_vectors(spot_peak_times_days, spot_widths_days)
        if peaks.size != latitudes.size or np.any(widths <= 0):
            raise ValueError("Provide one peak and positive evolution width per spot")
        evolution = np.exp(-0.5 * ((time_days[None, :] - peaks[:, None]) / widths[:, None]) ** 2)
    spot_flux = np.empty((latitudes.size, time_days.size))
    for index in range(latitudes.size):
        spot_flux[index] = _spot_light_curve(
            time_days, periods[index], longitudes[index], latitudes[index], radii[index],
            inclination_degrees, spot_contrast, limb_darkening, quadrature_order,
        )
    evolved_spot_flux = 1.0 - (1.0 - spot_flux) * evolution
    flare_peaks, amplitudes, rise_times, decay_times = _aligned_vectors(
        flare_peak_times_days, flare_amplitudes, flare_rise_times_days, flare_decay_times_days,
    )
    if np.any(amplitudes < 0) or np.any(rise_times <= 0) or np.any(decay_times <= 0):
        raise ValueError("Flare amplitudes must be nonnegative and time scales positive")
    flare_flux = np.zeros_like(time_days)
    for peak, amplitude, rise, decay in zip(flare_peaks, amplitudes, rise_times, decay_times):
        flare_flux += retr_lcurmodl_flarsing(time_days, peak, amplitude, rise, decay)
    flux = 1.0 + flux_offset + np.sum(evolved_spot_flux - 1.0, axis=0) + flare_flux
    if return_components:
        return {"flux": flux, "spot_flux": spot_flux,
                "evolved_spot_flux": evolved_spot_flux, "flare_flux": flare_flux}
    return flux