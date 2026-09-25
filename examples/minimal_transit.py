#!/usr/bin/env python3
"""Generate a deterministic Ephesos transit-model figure."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import ephesos


def evaluate_transit() -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a central transit under explicit dimensionless assumptions."""

    time_hours = np.linspace(-4.0, 4.0, 321)  # [hour]
    period_days = 3.0  # [day]
    radius_ratio = 0.1
    summed_radius_to_semimajor_axis = 0.1

    result = ephesos.eval_modl(
        time_hours / 24.0,
        "PlanetarySystem",
        pericomp=np.array([period_days]),
        epocmtracomp=np.array([0.0]),  # [day]
        rsmacomp=np.array([summed_radius_to_semimajor_axis]),
        cosicomp=np.array([0.0]),
        rratcomp=np.array([radius_ratio]),
        typelmdk="quad",
        booldiag=False,
        typeverb=0,
    )
    return time_hours, result["rflx"][:, 0]


def run_example(
    output_path: Path, animation_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the transit model and write its figure and animation."""

    time_hours, relative_flux = evaluate_transit()
    figure, axis = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    axis.set_facecolor("white")
    axis.plot(
        time_hours,
        relative_flux,
        color="#16697A",
        linewidth=2.3,
        label="Ephesos quadratic limb-darkened model",
    )
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Time from mid-transit [hour]")
    axis.set_ylabel("Relative flux")
    axis.set_title("Central transit for a companion-to-star radius ratio of 0.1")
    axis.grid(False)
    axis.legend(
        loc="lower right",
        frameon=True,
        fancybox=True,
        framealpha=1.0,
    )
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix == ".png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)

    if animation_path is None:
        animation_path = output_path.with_suffix(".gif")
    ephesos.save_light_curve_animation(
        time_hours,
        relative_flux,
        animation_path,
        title="Central transit",
        time_label="Time from mid-transit [hour]",
    )
    return time_hours, relative_flux


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic Ephesos transit-model figure."
    )
    parser.add_argument(
        "--typefileplot",
        choices=("png", "pdf"),
        default="png",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_path = Path(__file__).with_name(
        f"minimal_transit.{arguments.typefileplot}"
    )
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())