#!/usr/bin/env python3
"""Animate frequent transits from a compact seven-planet resonant chain."""

from pathlib import Path

import numpy as np

import ephesos
from ephesos.cli import run_output_example


PERIOD_DAYS = np.array((1.51, 2.42, 4.05, 6.10, 9.21, 12.35, 18.77))  # [day]
TRANSIT_EPOCH_DAYS = np.array((0.30, 1.10, 2.00, 3.00, 4.20, 5.40, 6.80))  # [day]
RADIUS_RATIO = np.array((0.070, 0.067, 0.075, 0.072, 0.079, 0.074, 0.081))
PLANET_MASS_EARTH = np.array((1.37, 1.31, 0.39, 0.69, 1.04, 1.32, 0.33))  # [Earth mass]
STELLAR_MASS_SOLAR = 0.089  # [solar mass]
STELLAR_RADIUS_SOLAR = 0.121  # [solar radius]


def evaluate_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a TRAPPIST-1-like chain and return its geometry and light curve."""

    # Kepler's third law gives a/R_star = 4.21 M_star^(1/3) P^(2/3) / R_star.
    semimajor_axis_stellar_radii = (
        4.21 * STELLAR_MASS_SOLAR ** (1.0 / 3.0) * PERIOD_DAYS ** (2.0 / 3.0) / STELLAR_RADIUS_SOLAR
    )
    summed_radius_to_semimajor_axis = (1.0 + RADIUS_RATIO) / semimajor_axis_stellar_radii
    impact_parameter = np.array((0.08, 0.13, 0.18, 0.24, 0.30, 0.37, 0.44))
    cosine_inclination = impact_parameter / semimajor_axis_stellar_radii
    time_days = np.linspace(0.0, 12.0, 8641)  # [day]
    relative_flux = ephesos.evaluate_multiplanet_transit_model(
        time_days,
        period_days=PERIOD_DAYS,
        transit_epoch_days=TRANSIT_EPOCH_DAYS,
        radius_ratio=RADIUS_RATIO,
        summed_radius_to_semimajor_axis=summed_radius_to_semimajor_axis,
        cosine_inclination=cosine_inclination,
    )
    hill_separations = ephesos.mutual_hill_separations(
        PERIOD_DAYS,
        PLANET_MASS_EARTH,
        STELLAR_MASS_SOLAR,
    )
    return time_days, relative_flux, summed_radius_to_semimajor_axis, hill_separations


def run_example(
    output_path: Path,
    max_frames: int = 144,
) -> tuple[np.ndarray, np.ndarray, tuple[Path, Path]]:
    """Write reveal and trailing variants of the compact-system animation."""

    time_days, relative_flux, summed_radius, hill_separations = evaluate_system()
    if np.min(hill_separations) <= 2.0 * np.sqrt(3.0):
        raise ValueError("adjacent planets must be separated by the pairwise Hill-stability limit")
    semimajor_axis_stellar_radii = (1.0 + RADIUS_RATIO) / summed_radius
    impact_parameter = np.array((0.08, 0.13, 0.18, 0.24, 0.30, 0.37, 0.44))
    output_stem = output_path.with_suffix("")
    output_paths = tuple(
        output_stem.with_name(f"{output_stem.name}_{mode}.gif") for mode in ("reveal", "trailing")
    )
    for mode, mode_output_path in zip(("reveal", "trailing"), output_paths):
        ephesos.save_light_curve_animation(
            time_days,
            relative_flux,
            mode_output_path,
            title="Compact seven-planet resonant chain",
            period=PERIOD_DAYS,
            transit_epoch=TRANSIT_EPOCH_DAYS,
            radius_ratio=RADIUS_RATIO,
            summed_radius_to_semimajor_axis=summed_radius,
            cosine_inclination=impact_parameter / semimajor_axis_stellar_radii,
            model_label="Combined seven-planet model",
            time_label="Time [day]",
            max_frames=max_frames,
            light_curve_mode=mode,
        )
    return time_days, relative_flux, output_paths


def main() -> int:
    return run_output_example(
        run_example,
        Path(__file__).with_suffix(".gif"),
        __doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
