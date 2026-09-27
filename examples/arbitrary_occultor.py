#!/usr/bin/env python3
"""Animate a deterministic transit by an extended ringed occultor."""

import argparse
from pathlib import Path

import numpy as np

import ephesos


def run_example(output_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate and animate the inclined-ring occultor light curve."""

    time_hours = np.linspace(-5.0, 5.0, 241)  # [hour]
    relative_flux = ephesos.evaluate_transit_model(
        time_hours / 24.0,
        period_days=4.0,  # [day]
        radius_ratio=0.09,
        summed_radius_to_semimajor_axis=0.12,
        cosine_inclination=0.03,
        system_type="PlanetarySystemWithRingsInclinedVertical",
    )
    ephesos.save_light_curve_animation(
        time_hours,
        relative_flux,
        output_path,
        title="Transit by an inclined ringed occultor",
        time_label="Time from mid-transit [hour]",
    )
    return time_hours, relative_flux


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".gif"))
    arguments = parser.parse_args()
    run_example(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
