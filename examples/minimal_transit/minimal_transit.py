#!/usr/bin/env python3
"""Generate a deterministic Ephesos transit-model figure."""

from pathlib import Path

import numpy as np

import ephesos
from tdpy.cli import parse_plot_arguments


PERIOD_DAYS = 3.0  # [day]
RADIUS_RATIO = 0.1
SUMMED_RADIUS_TO_SEMIMAJOR_AXIS = 0.1


def evaluate_transit() -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a central transit under explicit dimensionless assumptions."""

    time_hours = np.linspace(-4.0, 4.0, 321)  # [hour]
    relative_flux = ephesos.evaluate_transit_model(
        time_hours / 24.0,
        period_days=PERIOD_DAYS,
        radius_ratio=RADIUS_RATIO,
        summed_radius_to_semimajor_axis=SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
    )
    return time_hours, relative_flux


def run_example(
    output_path: Path, animation_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the transit model and write its figure and animation."""

    time_hours, relative_flux = evaluate_transit()
    typefileplot = output_path.suffix.removeprefix(".")
    ephesos.save_light_curve_figure(
        time_hours,
        relative_flux,
        output_path,
        title="Central transit for a companion-to-star radius ratio of 0.1",
        time_label="Time from mid-transit [hour]",
        model_label="Ephesos quadratic limb-darkened model",
        typefileplot=typefileplot,
    )

    if animation_path is None:
        animation_path = output_path.with_suffix(".gif")
    ephesos.save_light_curve_animation(
        time_hours,
        relative_flux,
        animation_path,
        title="Central transit",
        period=PERIOD_DAYS * 24.0,  # [hour]
        radius_ratio=RADIUS_RATIO,
        summed_radius_to_semimajor_axis=SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
        time_label="Time from mid-transit [hour]",
    )
    return time_hours, relative_flux


def parse_arguments():
    return parse_plot_arguments(
        description="Generate a deterministic Ephesos transit-model figure."
    )


def main() -> int:
    arguments = parse_arguments()
    output_path = Path(__file__).with_name(f"minimal_transit.{arguments.typefileplot}")
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
