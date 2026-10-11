"""Timing predictions for circular, coplanar exomoon systems."""

import numpy as np


def evaluate_exomoon_transit_time_offsets(
    *,
    transit_count: int,
    planet_period_days: float,
    planet_radius_ratio: float,
    summed_radius_to_semimajor_axis: float,
    moon_radius_ratio: float,
    moon_separation_planet_radii: float,
    moon_period_days: float,
    transit_epoch_days: float = 0.0,
    initial_moon_phase_radians: float = 0.0,
    cosine_inclination: float = 0.0,
) -> dict[str, np.ndarray]:
    """Predict moon-transit times relative to successive host-planet transits.

    The circular, coplanar moon phase advances deterministically between planet
    transits. Each offset is the moon's sky-plane x-axis crossing near the host
    transit, solved with Newton iterations. Moon transits hidden by the planet,
    behind the star, or outside the stellar disk are marked unobservable.
    ``initial_moon_phase_radians`` is the moon phase at ``transit_epoch_days``.
    """

    parameters = np.asarray(
        (
            planet_period_days,
            planet_radius_ratio,
            summed_radius_to_semimajor_axis,
            moon_radius_ratio,
            moon_separation_planet_radii,
            moon_period_days,
            transit_epoch_days,
            initial_moon_phase_radians,
            cosine_inclination,
        ),
        dtype=float,
    )
    if not np.isfinite(parameters).all():
        raise ValueError("exomoon timing parameters must be finite")
    if not isinstance(transit_count, (int, np.integer)) or transit_count < 2:
        raise ValueError("transit_count must be an integer of at least two")
    if planet_period_days <= 0.0 or moon_period_days <= 0.0:
        raise ValueError("planet_period_days and moon_period_days must be positive")
    if planet_radius_ratio <= 0.0 or moon_radius_ratio < 0.0:
        raise ValueError("planet radius must be positive and moon radius nonnegative")
    if summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if moon_separation_planet_radii <= 0.0:
        raise ValueError("moon_separation_planet_radii must be positive")
    if not -1.0 <= cosine_inclination <= 1.0:
        raise ValueError("cosine_inclination must be between -1 and 1")

    planet_semimajor_axis = (
        1.0 + planet_radius_ratio
    ) / summed_radius_to_semimajor_axis  # [R_star]
    moon_separation = moon_separation_planet_radii * planet_radius_ratio  # [R_star]
    planet_angular_frequency = 2.0 * np.pi / planet_period_days  # [rad day^-1]
    moon_angular_frequency = 2.0 * np.pi / moon_period_days  # [rad day^-1]
    planet_speed = planet_semimajor_axis * planet_angular_frequency  # [R_star day^-1]
    moon_speed = moon_separation * moon_angular_frequency  # [R_star day^-1]
    if moon_speed >= planet_speed:
        raise ValueError("moon orbital speed must be below the planet's transit speed")

    transit_number = np.arange(transit_count)
    planet_transit_times_days = transit_epoch_days + transit_number * planet_period_days
    moon_phase_at_transit = (
        initial_moon_phase_radians + transit_number * planet_period_days * moon_angular_frequency
    )  # [rad]
    time_offset_days = -moon_separation * np.sin(moon_phase_at_transit) / planet_speed
    for _ in range(12):
        planet_phase_at_moon_transit = planet_angular_frequency * time_offset_days
        moon_phase_at_moon_transit = (
            moon_phase_at_transit + moon_angular_frequency * time_offset_days
        )
        horizontal_position = planet_semimajor_axis * np.sin(
            planet_phase_at_moon_transit
        ) + moon_separation * np.sin(moon_phase_at_moon_transit)
        horizontal_velocity = planet_speed * np.cos(
            planet_phase_at_moon_transit
        ) + moon_speed * np.cos(moon_phase_at_moon_transit)
        correction_days = horizontal_position / horizontal_velocity  # [day]
        time_offset_days -= correction_days
        if np.all(np.abs(correction_days) < 1e-12):
            break
    else:
        raise RuntimeError("moon transit-time iteration did not converge")

    planet_phase_at_moon_transit = planet_angular_frequency * time_offset_days
    moon_phase_at_moon_transit = moon_phase_at_transit + moon_angular_frequency * time_offset_days
    inclination_sine = np.sqrt(1.0 - cosine_inclination**2)
    projected_y = cosine_inclination * (
        planet_semimajor_axis * np.cos(planet_phase_at_moon_transit)
        + moon_separation * np.cos(moon_phase_at_moon_transit)
    )  # [R_star]
    line_of_sight = inclination_sine * (
        planet_semimajor_axis * np.cos(planet_phase_at_moon_transit)
        + moon_separation * np.cos(moon_phase_at_moon_transit)
    )  # [R_star]
    moon_relative_x = moon_separation * np.sin(moon_phase_at_moon_transit)  # [R_star]
    moon_relative_y = (
        moon_separation * cosine_inclination * np.cos(moon_phase_at_moon_transit)
    )  # [R_star]
    moon_relative_z = (
        moon_separation * inclination_sine * np.cos(moon_phase_at_moon_transit)
    )  # [R_star]
    hidden_by_planet = (moon_relative_z < 0.0) & (
        moon_relative_x**2 + moon_relative_y**2 <= (planet_radius_ratio + moon_radius_ratio) ** 2
    )
    observable = (
        (line_of_sight > 0.0) & (np.abs(projected_y) <= 1.0 + moon_radius_ratio) & ~hidden_by_planet
    )
    relative_time_hours = np.where(observable, 24.0 * time_offset_days, np.nan)  # [hour]
    moon_transit_times_days = np.where(
        observable,
        planet_transit_times_days + time_offset_days,
        np.nan,
    )  # [day]

    return {
        "transit_number": transit_number,
        "planet_transit_times_days": planet_transit_times_days,
        "moon_transit_times_days": moon_transit_times_days,
        "relative_time_hours": relative_time_hours,
        "geometric_time_offset_hours": 24.0 * time_offset_days,
        "moon_phase_radians": np.mod(moon_phase_at_transit, 2.0 * np.pi),
        "observable": observable,
    }
