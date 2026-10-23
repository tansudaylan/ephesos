#!/usr/bin/env python3
"""Model an isolated planetary transit in a synthetic eclipsing-binary light curve."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tdpy.cli import add_plot_arguments
from tdpy.verbosity import print

import ephesos


BINARY_PERIOD_DAYS = 4.0  # [day]
BINARY_RADIUS_RATIO = 0.3
BINARY_SUMMED_RADIUS_TO_SEMIMAJOR_AXIS = 0.15
BINARY_SURFACE_BRIGHTNESS_RATIO = 0.5
PLANET_PERIOD_DAYS = 14.0  # [day]
PLANET_EPOCH_DAYS = 0.8  # [day]
PLANET_RADIUS_RATIO = 0.06
PLANET_SUMMED_RADIUS_TO_SEMIMAJOR_AXIS = 0.08
TIME_SPAN_DAYS = 30.0  # [day]
CADENCE_DAYS = 0.01  # [day]


def evaluate_system(time_days: np.ndarray) -> dict[str, np.ndarray]:
    """Combine native Ephesos binary and primary-transit flux models.

    The independently specified events do not overlap. The planet is assumed
    to transit the primary; binary reflex motion and three-body dynamics are
    not included.
    """

    time_days = np.asarray(time_days, dtype=float)
    if time_days.ndim != 1 or time_days.size < 3 or not np.isfinite(time_days).all():
        raise ValueError("time_days must be a finite one-dimensional array")
    if not np.all(np.diff(time_days) > 0.0):
        raise ValueError("time_days must be strictly increasing")

    binary_flux = ephesos.evaluate_emitting_companion_model(
        time_days,
        period_days=BINARY_PERIOD_DAYS,
        radius_ratio=BINARY_RADIUS_RATIO,
        summed_radius_to_semimajor_axis=BINARY_SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
        companion_brightness_ratio=BINARY_SURFACE_BRIGHTNESS_RATIO,
    )
    primary_transit_flux = ephesos.evaluate_transit_model(
        time_days - PLANET_EPOCH_DAYS,
        period_days=PLANET_PERIOD_DAYS,
        radius_ratio=PLANET_RADIUS_RATIO,
        summed_radius_to_semimajor_axis=PLANET_SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
    )
    binary_companion_flux_ratio = (
        BINARY_RADIUS_RATIO**2 * BINARY_SURFACE_BRIGHTNESS_RATIO
    )
    baseline_flux = 1.0 + binary_companion_flux_ratio
    binary_relative_flux = binary_flux / baseline_flux
    planet_flux_change = (primary_transit_flux - 1.0) / baseline_flux
    combined_relative_flux = binary_relative_flux + planet_flux_change
    return {
        "time_days": time_days,
        "binary_relative_flux": binary_relative_flux,
        "planet_flux_change": planet_flux_change,
        "combined_relative_flux": combined_relative_flux,
    }


def run_example(output_path: Path) -> tuple[dict[str, np.ndarray], Path]:
    """Generate the component curves and a two-panel comparison figure."""

    time_days = np.arange(0.0, TIME_SPAN_DAYS, CADENCE_DAYS)  # [day]
    curves = evaluate_system(time_days)
    output_path = Path(output_path)
    if output_path.suffix.lower() not in (".png", ".pdf"):
        raise ValueError("output_path must have a .png or .pdf suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    axes[0].plot(
        time_days,
        curves["combined_relative_flux"],
        color="#16697A",
        linewidth=1.0,
        label="Binary plus planet",
    )
    axes[0].plot(
        time_days,
        curves["binary_relative_flux"],
        color="#C73E1D",
        linewidth=0.8,
        alpha=0.8,
        label="Eclipsing binary",
    )
    axes[0].set_ylabel("Relative flux")
    axes[0].legend(frameon=True, fancybox=True, framealpha=1.0, facecolor="white")
    axes[1].plot(
        time_days,
        1e6 * curves["planet_flux_change"],
        color="#16697A",
        linewidth=1.0,
        label="Planet crossing the primary",
    )
    axes[1].set_xlabel("Time [day]")
    axes[1].set_ylabel("Planet signal [ppm]")
    axes[1].legend(frameon=True, fancybox=True, framealpha=1.0, facecolor="white")
    for axis in axes:
        axis.grid(False)
    figure.suptitle("Planet transits in a synthetic eclipsing-binary system", fontsize=11)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix.lower() == ".png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return curves, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).parent / "visuals" / "planet_in_eclipsing_binary.png",
    )
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    output_path = arguments.output_path.with_suffix(f".{arguments.typefileplot}")
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
