#!/usr/bin/env python3
"""Compare Kepler observations of known self-lensers with Ephesos models."""

from tdpy.verbosity import print

import argparse
from tdpy.cli import add_plot_arguments
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import ephesos


@dataclass(frozen=True)
class SystemConfiguration:
    """Published system parameters used by the local Ephesos approximation."""

    name: str
    data_file: str
    reference: str
    period_days: float  # [day]
    source_radius_solar: float  # [solar radius]
    source_mass_solar: float  # [solar mass]
    lens_mass_solar: float  # [solar mass]
    impact_parameter: float
    limb_darkening_coefficients: tuple[float, float]


SYSTEMS = (
    SystemConfiguration(
        name="KOI-3278",
        data_file="koi3278_kepler.csv",
        reference="Kruse and Agol (2014)",
        period_days=88.1805979,  # [day]
        source_radius_solar=0.96,  # [solar radius]
        source_mass_solar=1.0326,  # [solar mass]
        lens_mass_solar=0.6269,  # [solar mass]
        impact_parameter=0.7027,
        limb_darkening_coefficients=(0.45, 0.23),
    ),
    SystemConfiguration(
        name="KIC 8145411",
        data_file="kic8145411_kepler.csv",
        reference="Masuda et al. (2019)",
        period_days=455.826,  # [day]
        source_radius_solar=1.269,  # [solar radius]
        source_mass_solar=1.132,  # [solar mass]
        lens_mass_solar=0.200,  # [solar mass]
        impact_parameter=0.10,
        limb_darkening_coefficients=(0.59, 0.15),
    ),
)


def load_observations(
    data_directory: Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load compact, event-centered Kepler observations bundled with the example."""

    if data_directory is None:
        data_directory = Path(__file__).with_name("data")
    observations = {}
    for system in SYSTEMS:
        data_path = Path(data_directory) / system.data_file
        print(f"Reading from {data_path}...")
        table = np.loadtxt(data_path, delimiter=",", comments="#", skiprows=1)
        observations[system.name] = (table[:, 0], table[:, 1], table[:, 2])
    return observations


def evaluate_systems(
    data_directory: Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate one Ephesos pulse at the timestamps of each observed light curve."""

    observations = load_observations(data_directory)
    comparisons = {}
    for system in SYSTEMS:
        time_hours, relative_flux, relative_flux_error = observations[system.name]
        model_flux = ephesos.evaluate_self_lensing_model(
            time_hours / 24.0,
            period_days=system.period_days,
            source_radius_solar=system.source_radius_solar,
            source_mass_solar=system.source_mass_solar,
            lens_mass_solar=system.lens_mass_solar,
            impact_parameter=system.impact_parameter,
            limb_darkening_coefficients=system.limb_darkening_coefficients,
        )
        comparisons[system.name] = (
            time_hours,
            relative_flux,
            relative_flux_error,
            model_flux,
        )
    return comparisons


def run_example(
    output_path: Path,
    typefileplot: str = "png",
    data_directory: Path | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], Path]:
    """Plot known Kepler self-lensers and local Ephesos predictions."""

    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    comparisons = evaluate_systems(data_directory)
    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.3), sharex=False)
    for axis, system in zip(axes, SYSTEMS):
        time_hours, relative_flux, relative_flux_error, model_flux = comparisons[system.name]
        axis.errorbar(
            time_hours,
            1e6 * (relative_flux - 1.0),
            yerr=1e6 * relative_flux_error,
            color="#1F5673",
            marker="o",
            markersize=3.8,
            linestyle="none",
            linewidth=0.8,
            alpha=0.85,
            label=f"Kepler data, {system.reference}",
        )
        order = np.argsort(time_hours)
        axis.plot(
            time_hours[order],
            1e6 * (model_flux[order] - 1.0),
            color="#C73E1D",
            linewidth=2.2,
            label="Ephesos circular finite-source model",
        )
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.set_title(
            rf"{system.name}: $M_\mathrm{{lens}}={system.lens_mass_solar:.3g}\,M_\odot$, "
            rf"$P={system.period_days:.3g}$ d"
        )
        axis.set_ylabel("Flux excess [ppm]")
        axis.grid(False)
        axis.legend(loc="upper right", fancybox=True, framealpha=1.0, fontsize=10.0)
        axis.tick_params(labelsize=10.0)
        axis.title.set_fontsize(10.0)
        axis.xaxis.label.set_size(10.0)
        axis.yaxis.label.set_size(10.0)
    axes[-1].set_xlabel("Time from pulse center [hour]")
    figure.tight_layout()

    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if typefileplot == "png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return comparisons, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).with_name("known_self_lensers.png"),
    )
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    run_example(arguments.output_path, typefileplot=arguments.typefileplot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())