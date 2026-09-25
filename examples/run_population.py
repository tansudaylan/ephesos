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
        result = ephesos.eval_modl(
            time_hours / 24.0,
            "PlanetarySystem",
            pericomp=np.array([period_days]),  # [day]
            epocmtracomp=np.array([0.0]),  # [day]
            rsmacomp=np.array([0.1]),
            cosicomp=np.array([cosine_inclination]),
            rratcomp=np.array([radius_ratio]),
            typelmdk="quad",
            booldiag=False,
            typeverb=0,
        )
        output_path = output_directory / f"population_{name}.gif"
        ephesos.save_light_curve_animation(
            time_hours,
            result["rflx"][:, 0],
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