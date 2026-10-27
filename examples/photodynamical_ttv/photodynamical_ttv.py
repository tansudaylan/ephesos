#!/usr/bin/env python3
"""Predict transit-time variations in a near-resonant planetary system."""

import argparse
from pathlib import Path

from tdpy.cli import add_plot_arguments
from tdpy.verbosity import print

import matplotlib.pyplot as plt
import numpy as np

import ephesos


STAR_MASS_SOLAR = 0.8  # [solar mass]
START_TIME_DAYS = 0.0  # [day]
STEP_DAYS = 0.05  # [day]
END_TIME_DAYS = 500.0  # [day]
PLANET_PARAMETERS = np.array(
    (
        (3.0e-5, 10.0, 0.01, 89.8, 0.0, 90.0, 0.0),
        (2.5e-5, 15.1, 0.02, 89.9, 0.0, 90.0, 15.0),
    ),
    dtype=float,
)  # [solar mass, day, dimensionless, degree, degree, degree, degree]


def calculate_linear_residuals(
    predictions: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fit and subtract a linear ephemeris from each planet's event times."""

    residuals = []
    for transit_epochs, transit_times_days in predictions:
        transit_epochs = np.asarray(transit_epochs, dtype=float)
        transit_times_days = np.asarray(transit_times_days, dtype=float)
        if (
            transit_epochs.ndim != 1
            or transit_times_days.shape != transit_epochs.shape
            or transit_epochs.size < 2
            or not np.isfinite(transit_epochs).all()
            or not np.isfinite(transit_times_days).all()
        ):
            raise ValueError("each planet requires at least two finite transit epochs and times")
        fitted_period_days, fitted_epoch_days = np.polyfit(transit_epochs, transit_times_days, 1)
        linear_times = ephesos.evaluate_linear_transit_times(
            transit_epochs,
            fitted_epoch_days,
            fitted_period_days,
        )  # [day]
        residuals.append((transit_epochs, 1440.0 * (transit_times_days - linear_times)))  # [minute]
    return residuals


def run_example(output_path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray]], Path]:
    """Integrate a two-planet system and plot its predicted O-C timings."""

    predictions = ephesos.evaluate_nbody_transit_times(
        PLANET_PARAMETERS,
        STAR_MASS_SOLAR,
        START_TIME_DAYS,
        STEP_DAYS,
        END_TIME_DAYS,
    )
    residuals = calculate_linear_residuals(predictions)
    output_path = Path(output_path)
    if output_path.suffix.lower() not in (".png", ".pdf"):
        raise ValueError("output_path must have a .png or .pdf suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        len(residuals),
        1,
        figsize=(7.2, 5.6),
        sharex=False,
        squeeze=False,
    )
    for planet_index, (axis, (transit_epochs, residual_minutes)) in enumerate(
        zip(axes[:, 0], residuals)
    ):
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.plot(
            transit_epochs,
            residual_minutes,
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            color=("#16697A", "#C73E1D")[planet_index % 2],
        )
        axis.set_ylabel("O-C [minute]")
        axis.set_title(f"Planet {planet_index + 1}, P={PLANET_PARAMETERS[planet_index, 1]:g} d")
        axis.grid(False)
    axes[-1, 0].set_xlabel("Transit epoch")
    figure.suptitle(
        "N-body transit-time variations in a near-3:2 planetary pair",
        fontsize=12.0,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix.lower() == ".png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return residuals, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).parent / "visuals" / "photodynamical_ttv.png",
    )
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    output_path = arguments.output_path.with_suffix(f".{arguments.typefileplot}")
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
