#!/usr/bin/env python3
"""Animate a planet transiting a compact white dwarf."""

import argparse
from pathlib import Path

import numpy as np

import ephesos


def run_example(output_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate and animate a deep white-dwarf transit."""

    time_minutes = np.linspace(-12.0, 12.0, 241)  # [minute]
    relative_flux = ephesos.evaluate_transit_model(
        time_minutes / 1440.0,
        period_days=0.8,  # [day]
        radius_ratio=0.55,
        summed_radius_to_semimajor_axis=0.015,
    )
    ephesos.save_light_curve_animation(
        time_minutes,
        relative_flux,
        output_path,
        title="Planet transiting a white dwarf",
        time_label="Time from mid-transit [minute]",
    )
    return time_minutes, relative_flux


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".gif"))
    arguments = parser.parse_args()
    run_example(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
