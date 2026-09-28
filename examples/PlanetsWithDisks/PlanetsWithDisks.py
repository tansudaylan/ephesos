#!/usr/bin/env python3
"""Compare equal-area circumplanetary-disk transit morphologies."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import ephesos


TIME_HOURS = np.linspace(-5.0, 5.0, 401)  # [hour]
PERIOD_DAYS = 10.0  # [day]
EQUIVALENT_RADIUS_RATIO = 0.09
SUMMED_RADIUS_TO_SEMIMAJOR_AXIS = 0.08
COSINE_INCLINATION = 0.02
MORPHOLOGIES = {
    "disk": "Spherical planet",
    "face_on_rings": "Face-on disk",
    "horizontal_rings": "Edge-on horizontal disk",
    "vertical_rings": "Edge-on vertical disk",
}


def evaluate_models() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate equal-area planet and circumplanetary-disk silhouettes."""

    model_arguments = {
        "period_days": PERIOD_DAYS,
        "equivalent_radius_ratio": EQUIVALENT_RADIUS_RATIO,
        "summed_radius_to_semimajor_axis": SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
        "cosine_inclination": COSINE_INCLINATION,
    }
    relative_flux = {
        occultor_type: ephesos.evaluate_projected_occultor_model(
            TIME_HOURS / 24.0,
            occultor_type=occultor_type,
            **model_arguments,
        )
        for occultor_type in MORPHOLOGIES
    }
    return TIME_HOURS, relative_flux


def run_example(
    output_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], Path]:
    """Plot disk light curves and departures from the spherical benchmark."""

    time_hours, relative_flux = evaluate_models()
    output_path = Path(output_path)
    if output_path.suffix.lower() not in (".png", ".pdf"):
        raise ValueError("output_path must have a .png or .pdf suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, (flux_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    colors = ("black", "#16697A", "#C73E1D", "#2D7D46")
    benchmark_flux = relative_flux["disk"]
    for color, (occultor_type, label) in zip(colors, MORPHOLOGIES.items()):
        flux_axis.plot(time_hours, relative_flux[occultor_type], color=color, label=label)
        if occultor_type != "disk":
            residual_axis.plot(
                time_hours,
                1e3 * (relative_flux[occultor_type] - benchmark_flux),
                color=color,
                label=label,
            )
    flux_axis.set_ylabel("Relative flux")
    flux_axis.set_title("Equal-area circumplanetary-disk transits")
    residual_axis.set_xlabel("Time from mid-transit [hour]")
    residual_axis.set_ylabel("Departure [ppt]")
    for axis in (flux_axis, residual_axis):
        axis.grid(False)
    flux_axis.legend(loc="lower right", fancybox=True, framealpha=1.0)
    residual_axis.axhline(0.0, color="black", linewidth=1.0)
    figure.tight_layout()

    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix.lower() == ".png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return time_hours, relative_flux, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).with_name("PlanetsWithDisks.png"),
    )
    arguments = parser.parse_args()
    run_example(arguments.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
