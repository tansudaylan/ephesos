#!/usr/bin/env python3
"""Animate representative transits from a deterministic planet population."""

import argparse
from pathlib import Path

import numpy as np

import ephesos


def run_example(output_directory: Path) -> list[Path]:
    """Evaluate three systems and write one animation for each light curve."""

    output_directory.mkdir(parents=True, exist_ok=True)
    time_hours = np.linspace(-5.0, 5.0, 241)  # [hour]
    systems = (
        ("compact", 2.0, 0.07, 0.02),
        ("warm", 5.0, 0.10, 0.08),
        ("grazing", 8.0, 0.13, 0.15),
    )
    output_paths = []
    for name, period_days, radius_ratio, cosine_inclination in systems:
        relative_flux = ephesos.evaluate_transit_model(
            time_hours / 24.0,
            period_days=period_days,  # [day]
            radius_ratio=radius_ratio,
            summed_radius_to_semimajor_axis=0.1,
            cosine_inclination=cosine_inclination,
        )
        output_path = output_directory / f"population_{name}.gif"
        ephesos.save_light_curve_animation(
            time_hours,
            relative_flux,
            output_path,
            title=f"Population member: {name}",
            time_label="Time from mid-transit [hour]",
        )
        output_paths.append(output_path)
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).parent / "population_animations",
    )
    arguments = parser.parse_args()
    run_example(arguments.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
