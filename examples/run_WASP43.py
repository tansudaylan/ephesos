#!/usr/bin/env python3
"""Animate an Ephesos model of the WASP-43 b light curve."""

import argparse
from pathlib import Path

import numpy as np

import ephesos


def run_example(output_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate and animate the published WASP-43 b geometry."""

    time_hours = np.linspace(-3.0, 3.0, 241)  # [hour]
    result = ephesos.eval_modl(
        time_hours / 24.0,
        "PlanetarySystem",
        pericomp=np.array([0.813475]),  # [day]
        epocmtracomp=np.array([0.0]),  # [day]
        rsmacomp=np.array([0.21]),
        cosicomp=np.array([0.134]),
        rratcomp=np.array([0.1615]),
        coeflmdk=np.array([0.1, 0.05]),
        typelmdk="quad",
        booldiag=False,
        typeverb=0,
    )
    relative_flux = result["rflx"][:, 0]
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