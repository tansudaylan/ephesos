#!/usr/bin/env python3
"""Demonstrate Rømer's Io-eclipse light-time argument with synthetic timings."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdpy.cli import add_plot_arguments
from tdpy.verbosity import print


AU_KM = 149_597_870.7  # [km]
SECONDS_PER_DAY = 86_400.0  # [s day^-1]
SECONDS_PER_MINUTE = 60.0  # [s min^-1]
MINUTES_PER_DAY = 1_440.0  # [min day^-1]
LIGHT_SPEED_KM_S = 299_792.458  # [km s^-1]
EARTH_SEMIMAJOR_AXIS_AU = 1.00000011  # [AU]
EARTH_PERIOD_DAYS = 365.256  # [day]
JUPITER_SEMIMAJOR_AXIS_AU = 5.20336301  # [AU]
JUPITER_PERIOD_DAYS = 4_332.589  # [day]
JUPITER_RADIUS_KM = 69_911.0  # [km]
IO_ORBIT_RADIUS_KM = 421_800.0  # [km]
IO_RADIUS_KM = 1_821.5  # [km]
IO_PERIOD_DAYS = 1.769138  # [day]
SIMULATION_DURATION_DAYS = 220.0  # [day]
TIMING_UNCERTAINTY_SECONDS = 30.0  # [s]


def earth_jupiter_distance_au(time_days: np.ndarray) -> np.ndarray:
    """Return circular, coplanar Earth-Jupiter distances, with opposition at zero."""

    time_days = np.asarray(time_days, dtype=float)
    earth_phase = 2.0 * np.pi * time_days / EARTH_PERIOD_DAYS  # [rad]
    jupiter_phase = 2.0 * np.pi * time_days / JUPITER_PERIOD_DAYS  # [rad]
    distance_squared_au = (
        EARTH_SEMIMAJOR_AXIS_AU**2
        + JUPITER_SEMIMAJOR_AXIS_AU**2
        - 2.0
        * EARTH_SEMIMAJOR_AXIS_AU
        * JUPITER_SEMIMAJOR_AXIS_AU
        * np.cos(earth_phase - jupiter_phase)
    )  # [AU^2]
    return np.sqrt(distance_squared_au)  # [AU]


def estimate_light_speed(
    distance_au: np.ndarray,
    observed_minus_ephemeris_minutes: np.ndarray,
) -> dict[str, float]:
    """Fit the light-time slope and return the speed of light in km/s."""

    distance_au = np.asarray(distance_au, dtype=float)
    observed_minus_ephemeris_minutes = np.asarray(
        observed_minus_ephemeris_minutes, dtype=float
    )  # [min]
    if (
        distance_au.ndim != 1
        or observed_minus_ephemeris_minutes.shape != distance_au.shape
        or distance_au.size < 3
        or not np.isfinite(distance_au).all()
        or not np.isfinite(observed_minus_ephemeris_minutes).all()
    ):
        raise ValueError("distance and timing arrays must be matching finite vectors")
    if np.ptp(distance_au) <= 0.0:
        raise ValueError("distance_au must span a nonzero range")

    design = np.column_stack((distance_au, np.ones(distance_au.shape)))
    slope, intercept = np.linalg.lstsq(
        design, observed_minus_ephemeris_minutes, rcond=None
    )[0]  # [min AU^-1], [min]
    if slope <= 0.0:
        raise ValueError("the fitted light-time slope must be positive")
    residuals = observed_minus_ephemeris_minutes - design @ np.array((slope, intercept))  # [min]
    residual_variance = np.sum(residuals**2) / (distance_au.size - 2)  # [min^2]
    slope_uncertainty = np.sqrt(
        residual_variance * np.linalg.inv(design.T @ design)[0, 0]
    )  # [min AU^-1]
    speed_km_s = AU_KM / (slope * SECONDS_PER_MINUTE)  # [km s^-1]
    speed_uncertainty_km_s = speed_km_s * slope_uncertainty / slope  # [km s^-1]
    return {
        "slope_minutes_per_au": float(slope),
        "intercept_minutes": float(intercept),
        "speed_km_s": float(speed_km_s),
        "speed_uncertainty_km_s": float(speed_uncertainty_km_s),
    }


def make_synthetic_eclipse_observations(seed: int = 17) -> dict[str, np.ndarray | float]:
    """Generate explicitly synthetic Io immersion and emergence timings."""

    eclipse_centers_days = np.arange(
        -0.5 * SIMULATION_DURATION_DAYS,
        0.5 * SIMULATION_DURATION_DAYS,
        IO_PERIOD_DAYS,
    )  # [day]
    io_speed_km_s = 2.0 * np.pi * IO_ORBIT_RADIUS_KM / (
        IO_PERIOD_DAYS * SECONDS_PER_DAY
    )  # [km s^-1]
    eclipse_duration_days = (
        2.0 * (JUPITER_RADIUS_KM + IO_RADIUS_KM)
        / io_speed_km_s
        / SECONDS_PER_DAY
    )  # [day]
    true_contact_times_days = np.column_stack(
        (
            eclipse_centers_days - 0.5 * eclipse_duration_days,
            eclipse_centers_days + 0.5 * eclipse_duration_days,
        )
    ).ravel()  # [day]
    distance_at_centers_au = earth_jupiter_distance_au(eclipse_centers_days)  # [AU]
    light_time_days = (
        distance_at_centers_au * AU_KM / LIGHT_SPEED_KM_S / SECONDS_PER_DAY
    )  # [day]
    repeated_light_time_days = np.repeat(light_time_days, 2)  # [day]
    random_generator = np.random.default_rng(seed)
    timing_noise_days = random_generator.normal(
        0.0, TIMING_UNCERTAINTY_SECONDS / SECONDS_PER_DAY, true_contact_times_days.size
    )  # [day]
    observed_contact_times_days = (
        true_contact_times_days + repeated_light_time_days + timing_noise_days
    )  # [day]
    observed_minus_ephemeris_minutes = (
        observed_contact_times_days - true_contact_times_days
    ) * MINUTES_PER_DAY  # [min]
    repeated_distances_au = np.repeat(distance_at_centers_au, 2)  # [AU]
    fit = estimate_light_speed(repeated_distances_au, observed_minus_ephemeris_minutes)

    return {
        "eclipse_centers_days": eclipse_centers_days,
        "contact_times_days": observed_contact_times_days,
        "contact_kinds": np.tile(("Immersion", "Emergence"), eclipse_centers_days.size),
        "distance_au": repeated_distances_au,
        "observed_minus_ephemeris_minutes": observed_minus_ephemeris_minutes,
        "eclipse_duration_hours": 24.0 * eclipse_duration_days,  # [hour]
        **fit,
    }


def run_example(output_path: Path) -> tuple[dict[str, np.ndarray | float], Path]:
    """Plot synthetic eclipse timing delays and the fitted light-time slope."""

    observations = make_synthetic_eclipse_observations()
    output_path = Path(output_path)
    if output_path.suffix.lower() not in (".png", ".pdf"):
        raise ValueError("output_path must have a .png or .pdf suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    distance_au = np.asarray(observations["distance_au"], dtype=float)  # [AU]
    delay_minutes = np.asarray(
        observations["observed_minus_ephemeris_minutes"], dtype=float
    )  # [min]
    contact_kinds = np.asarray(observations["contact_kinds"])
    color_by_kind = {"Immersion": "#16697A", "Emergence": "#C73E1D"}
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.4))
    for kind, color in color_by_kind.items():
        selected = contact_kinds == kind
        axes[0].scatter(
            np.repeat(observations["eclipse_centers_days"], 2)[selected],
            delay_minutes[selected],
            color=color,
            s=17,
            label=kind,
        )
        axes[1].scatter(
            distance_au[selected],
            delay_minutes[selected],
            color=color,
            s=17,
            label=kind,
        )

    distance_grid_au = np.linspace(np.min(distance_au), np.max(distance_au), 100)  # [AU]
    fitted_delay_minutes = (
        observations["slope_minutes_per_au"] * distance_grid_au
        + observations["intercept_minutes"]
    )  # [min]
    axes[1].plot(
        distance_grid_au,
        fitted_delay_minutes,
        color="black",
        linewidth=1.2,
        label="Fitted light-time relation",
    )
    axes[0].set_xlabel("Time since opposition [day]")
    axes[0].set_ylabel("O-C [minute]")
    axes[1].set_xlabel("Earth-Jupiter distance [AU]")
    axes[1].set_ylabel("O-C [minute]")
    for axis in axes:
        axis.grid(False)
        axis.legend(frameon=True, fancybox=True, framealpha=1.0, facecolor="white")
    figure.suptitle(
        "Synthetic Io eclipse timings recover "
        f"c = {observations['speed_km_s']:.0f} +/- "
        f"{observations['speed_uncertainty_km_s']:.0f} km/s",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    print(
        "Recovered c = "
        f"{observations['speed_km_s']:.0f} +/- "
        f"{observations['speed_uncertainty_km_s']:.0f} km/s"
    )
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix.lower() == ".png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return observations, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).parent / "visuals" / "romer_light_time.png",
    )
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    output_path = arguments.output_path.with_suffix(f".{arguments.typefileplot}")
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
