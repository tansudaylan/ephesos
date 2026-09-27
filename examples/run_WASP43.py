#!/usr/bin/env python3
"""Animate an Ephesos model of the WASP-43 b light curve."""

import argparse
from pathlib import Path

import numpy as np

import ephesos


def run_example(output_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate and animate the published WASP-43 b geometry."""

    time_hours = np.linspace(-3.0, 3.0, 241)  # [hour]
    relative_flux = ephesos.evaluate_transit_model(
        time_hours / 24.0,
        period_days=0.813475,  # [day]
        radius_ratio=0.1615,
        summed_radius_to_semimajor_axis=0.21,
        cosine_inclination=0.134,
        limb_darkening_coefficients=np.array([0.1, 0.05]),
    )
    ephesos.save_light_curve_animation(
        time_hours,
        relative_flux,
        output_path,
        title="WASP-43 b transit model",
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
