#!/usr/bin/env python3
"""Animate a planet transiting a compact white dwarf."""

from pathlib import Path

import numpy as np

import ephesos
from ephesos.cli import run_output_example


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
    return run_output_example(run_example, Path(__file__).with_suffix(".gif"), __doc__)


if __name__ == "__main__":
    raise SystemExit(main())
