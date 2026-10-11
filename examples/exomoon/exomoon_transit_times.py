#!/usr/bin/env python3
"""Predict an exomoon's transit-time distribution across host-planet transits."""

import argparse
import csv
from pathlib import Path

from tdpy.cli import add_plot_arguments
from tdpy.verbosity import print

import matplotlib.pyplot as plt
import numpy as np

import ephesos


PLANET_PERIOD_DAYS = 4.0  # [day]
PLANET_RADIUS_RATIO = 0.09
SUMMED_RADIUS_TO_SEMIMAJOR_AXIS = 0.1
MOON_RADIUS_RATIO = 0.025
MOON_SEPARATION_PLANET_RADII = 5.0
MOON_PERIOD_DAYS = 0.47  # [day]
TRANSIT_COUNT = 128
TRANSIT_EPOCH_DAYS = 0.0  # [day]
INITIAL_MOON_PHASE_RADIANS = 0.4  # [rad]
COSINE_INCLINATION = 0.015


def run_pipeline(
    output_path: Path,
    data_path: Path,
    typefileplot: str = "png",
) -> tuple[dict[str, np.ndarray], Path, Path]:
    """Write predicted moon-transit times and a timing-distribution figure."""

    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    predictions = ephesos.evaluate_exomoon_transit_time_offsets(
        transit_count=TRANSIT_COUNT,
        planet_period_days=PLANET_PERIOD_DAYS,
        planet_radius_ratio=PLANET_RADIUS_RATIO,
        summed_radius_to_semimajor_axis=SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
        moon_radius_ratio=MOON_RADIUS_RATIO,
        moon_separation_planet_radii=MOON_SEPARATION_PLANET_RADII,
        moon_period_days=MOON_PERIOD_DAYS,
        transit_epoch_days=TRANSIT_EPOCH_DAYS,
        initial_moon_phase_radians=INITIAL_MOON_PHASE_RADIANS,
        cosine_inclination=COSINE_INCLINATION,
    )
    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    data_path = Path(data_path).with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    columns = (
        "transit_number",
        "planet_transit_times_days",
        "moon_transit_times_days",
        "relative_time_hours",
        "geometric_time_offset_hours",
        "moon_phase_radians",
        "observable",
    )
    print(f"Writing to {data_path}...")
    with data_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for row_index in range(TRANSIT_COUNT):
            writer.writerow({column: predictions[column][row_index] for column in columns})

    transit_number = predictions["transit_number"]
    observable = predictions["observable"]
    geometric_offsets = predictions["geometric_time_offset_hours"]
    observed_offsets = predictions["relative_time_hours"][observable]
    if observed_offsets.size == 0:
        raise ValueError("the selected system has no observable moon transits")
    moon_phase = predictions["moon_phase_radians"]

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), facecolor="white")
    axes[0].scatter(
        transit_number[observable],
        geometric_offsets[observable],
        color="#16697A",
        s=24,
        label="Observable moon transit",
    )
    axes[0].scatter(
        transit_number[~observable],
        geometric_offsets[~observable],
        color="#C73E1D",
        marker="x",
        s=28,
        label="Not observable",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Planet transit number")
    axes[0].set_ylabel("Moon offset [hour]")
    axes[0].set_title("Predicted offset by epoch")
    axes[0].legend(loc="upper right", fancybox=True, framealpha=1.0, fontsize=9.0)

    axes[1].hist(
        observed_offsets,
        bins=18,
        density=True,
        color="#C73E1D",
        alpha=0.85,
    )
    axes[1].set_xlabel("Moon offset [hour]")
    axes[1].set_ylabel(r"Probability density [hour$^{-1}$]")
    axes[1].set_title("Timing-offset distribution")

    axes[2].scatter(
        moon_phase[observable],
        geometric_offsets[observable],
        color="#2D7D46",
        s=24,
    )
    axes[2].set_xlabel("Moon phase at planet transit [rad]")
    axes[2].set_ylabel("Moon offset [hour]")
    axes[2].set_title("Moon phase sets the offset")
    for axis in axes:
        axis.grid(False)
    figure.suptitle(
        f"Pp={PLANET_PERIOD_DAYS:g} d, Pm={MOON_PERIOD_DAYS:g} d, "
        f"am/Rp={MOON_SEPARATION_PLANET_RADII:g}, "
        f"Rp/Rstar={PLANET_RADIUS_RATIO:g}",
        fontsize=11.0,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))

    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if typefileplot == "png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return predictions, output_path, data_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).parent / "visuals" / "exomoon_transit_times.png",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).parent / "visuals" / "exomoon_transit_times.csv",
    )
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    run_pipeline(
        arguments.output_path,
        arguments.data_path,
        typefileplot=arguments.typefileplot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
