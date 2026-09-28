#!/usr/bin/env python3
"""Animate two planets transiting simultaneously on resonant orbits."""

import argparse
from pathlib import Path

import numpy as np

import ephesos


TIME_HOURS = np.linspace(-5.0, 5.0, 401)  # [hour]
PERIOD_DAYS = np.array((2.0, 3.0))  # [day]
TRANSIT_EPOCH_DAYS = np.zeros(2)  # [day]
RADIUS_RATIO = np.array((0.08, 0.06))
IMPACT_PARAMETER = np.array((0.15, 0.35))


def evaluate_system() -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Evaluate simultaneous combined and individual transit light curves."""

    semimajor_axis_stellar_radii = 4.21 * PERIOD_DAYS ** (2.0 / 3.0)
    summed_radius_to_semimajor_axis = (1.0 + RADIUS_RATIO) / semimajor_axis_stellar_radii
    cosine_inclination = IMPACT_PARAMETER / semimajor_axis_stellar_radii
    model_arguments = {
        "period_days": PERIOD_DAYS,
        "transit_epoch_days": TRANSIT_EPOCH_DAYS,
        "radius_ratio": RADIUS_RATIO,
        "summed_radius_to_semimajor_axis": summed_radius_to_semimajor_axis,
        "cosine_inclination": cosine_inclination,
    }
    combined_flux = ephesos.evaluate_multiplanet_transit_model(
        TIME_HOURS / 24.0,
        **model_arguments,
    )
    individual_flux = tuple(
        ephesos.evaluate_transit_model(
            TIME_HOURS / 24.0,
            period_days=PERIOD_DAYS[index],
            radius_ratio=RADIUS_RATIO[index],
            summed_radius_to_semimajor_axis=summed_radius_to_semimajor_axis[index],
            cosine_inclination=cosine_inclination[index],
        )
        for index in range(2)
    )
    return TIME_HOURS, combined_flux, individual_flux, summed_radius_to_semimajor_axis


def run_example(
    output_path: Path,
    max_frames: int = 72,
) -> tuple[np.ndarray, np.ndarray]:
    """Write a synchronized simultaneous-transit animation."""

    time_hours, combined_flux, individual_flux, summed_radius = evaluate_system()
    semimajor_axis_stellar_radii = (1.0 + RADIUS_RATIO) / summed_radius
    ephesos.save_light_curve_animation(
        time_hours,
        combined_flux,
        output_path,
        title="Simultaneous transits in a 3:2 resonant pair",
        period=PERIOD_DAYS * 24.0,  # [hour]
        transit_epoch=TRANSIT_EPOCH_DAYS * 24.0,  # [hour]
        radius_ratio=RADIUS_RATIO,
        summed_radius_to_semimajor_axis=summed_radius,
        cosine_inclination=IMPACT_PARAMETER / semimajor_axis_stellar_radii,
        comparison_models={
            "Inner planet alone": individual_flux[0],
            "Outer planet alone": individual_flux[1],
        },
        model_label="Combined two-planet model",
        time_label="Time from shared mid-transit [hour]",
        max_frames=max_frames,
        simultaneous_transit_slowdown=3,
    )
    return time_hours, combined_flux


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).with_name("transits_simultaneous.gif"),
    )
    arguments = parser.parse_args()
    run_example(arguments.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
