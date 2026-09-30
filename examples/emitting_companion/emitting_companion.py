#!/usr/bin/env python3
"""Compare secondary eclipses from emitting planetary companions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ephesos


PERIOD_DAYS = 1.0  # [day]
RADIUS_RATIO = 0.1
SUMMED_RADIUS_TO_SEMIMAJOR_AXIS = 0.2
BRIGHTNESS_RATIOS = {
    "Cool companion": 0.002,
    "Warm companion": 0.005,
    "Hot companion": 0.010,
}


def evaluate_systems() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate planetary emission around secondary eclipse."""

    orbital_phase = np.linspace(0.25, 0.75, 401)
    time_days = orbital_phase * PERIOD_DAYS  # [day]
    relative_flux = {
        label: ephesos.evaluate_emitting_companion_model(
            time_days,
            period_days=PERIOD_DAYS,
            radius_ratio=RADIUS_RATIO,
            summed_radius_to_semimajor_axis=SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
            companion_brightness_ratio=brightness_ratio,
        )
        for label, brightness_ratio in BRIGHTNESS_RATIOS.items()
    }
    return orbital_phase, relative_flux


def run_example(
    output_path: Path,
    typefileplot: str = "png",
) -> tuple[np.ndarray, dict[str, np.ndarray], Path]:
    """Plot how planetary surface brightness controls eclipse depth."""

    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    orbital_phase, relative_flux = evaluate_systems()
    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    colors = ("#1F5673", "#2D7D46", "#C73E1D")
    for color, (label, flux) in zip(colors, relative_flux.items()):
        axis.plot(
            orbital_phase,
            1.0e6 * (flux - 1.0),
            color=color,
            linewidth=2.0,
            label=rf"{label}, $I_p/I_\star={BRIGHTNESS_RATIOS[label]:.3f}$",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Orbital phase")
    axis.set_ylabel("Flux excess [ppm]")
    axis.set_title("Planetary emission sets secondary-eclipse depth")
    axis.grid(False)
    axis.legend(loc="lower left", fancybox=True, framealpha=1.0)
    figure.tight_layout()

    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if typefileplot == "png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return orbital_phase, relative_flux, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()
    output_path = Path(__file__).with_name("visuals") / "emitting_companion.png"
    run_example(output_path, typefileplot=arguments.typefileplot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())