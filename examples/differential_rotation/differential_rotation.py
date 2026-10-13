#!/usr/bin/env python3
"""Visualize a clearly labeled simulation of stellar differential rotation."""

from tdpy.verbosity import print

from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import ephesos
from tdpy.cli import parse_plot_arguments


EQUATORIAL_PERIOD_DAYS = 10.0  # [day]
RELATIVE_SHEAR = 0.2
SIMULATION_DURATION_DAYS = 30.0  # [day]


def evaluate_differential_rotation() -> dict[str, np.ndarray]:
    """Evaluate the rotation law and the resulting longitudinal drift."""

    latitude_degrees = np.linspace(-90.0, 90.0, 181)  # [deg]
    rotation_period_days = ephesos.retr_protlati(
        EQUATORIAL_PERIOD_DAYS,
        RELATIVE_SHEAR,
        latitude_degrees,
    )
    time_days = np.linspace(0.0, SIMULATION_DURATION_DAYS, 301)  # [day]
    selected_latitude_degrees = np.array([0.0, 30.0, 60.0])  # [deg]
    selected_period_days = ephesos.retr_protlati(
        EQUATORIAL_PERIOD_DAYS,
        RELATIVE_SHEAR,
        selected_latitude_degrees,
    )
    longitude_lag_degrees = 360.0 * time_days[:, None] * (
        1.0 / EQUATORIAL_PERIOD_DAYS - 1.0 / selected_period_days[None, :]
    )  # [deg]
    return {
        "latitude_degrees": latitude_degrees,
        "rotation_period_days": rotation_period_days,
        "time_days": time_days,
        "selected_latitude_degrees": selected_latitude_degrees,
        "longitude_lag_degrees": longitude_lag_degrees,
    }


def run_example(output_path: Path) -> dict[str, np.ndarray]:
    """Plot the assumed rotation law and its predicted spot-phase drift."""

    products = evaluate_differential_rotation()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.4),
        facecolor="white",
        constrained_layout=True,
    )
    axes[0].plot(
        products["latitude_degrees"],
        products["rotation_period_days"],
        color="#1B6CA8",
        linewidth=2.2,
    )
    axes[0].set_xlabel("Spot latitude [deg]")
    axes[0].set_ylabel("Rotation period [day]")
    axes[0].set_title("Latitude-dependent rotation")

    for index, latitude_degrees in enumerate(
        products["selected_latitude_degrees"]
    ):
        axes[1].plot(
            products["time_days"],
            products["longitude_lag_degrees"][:, index],
            linewidth=2.0,
            label=rf"Latitude $={latitude_degrees:.0f}^\circ$",
        )
    axes[1].set_xlabel("Elapsed time [day]")
    axes[1].set_ylabel("Longitude lag behind equator [deg]")
    axes[1].set_title("Predicted spot-phase separation")
    legend = axes[1].legend(frameon=True, fancybox=True, framealpha=1.0)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    for axis in axes:
        axis.grid(False)
    figure.suptitle(
        "Simulated differential rotation with "
        f"$P_{{eq}}={EQUATORIAL_PERIOD_DAYS:.0f}$ day and "
        rf"$\rho={RELATIVE_SHEAR:.1f}$",
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix == ".png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return products


def parse_arguments():
    return parse_plot_arguments(
        description="Plot Ephesos's simulated differential-rotation prediction."
    )


def main() -> int:
    arguments = parse_arguments()
    output_path = Path(__file__).with_name(
        f"differential_rotation.{arguments.typefileplot}"
    )
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())