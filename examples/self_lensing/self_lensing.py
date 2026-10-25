#!/usr/bin/env python3
"""Compare toy self-lensing systems and finite-source effects."""

from tdpy.verbosity import print

import argparse
from datetime import datetime, timezone
from tdpy.cli import add_plot_arguments
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import ephesos


TIME_HOURS = np.linspace(-12.0, 12.0, 321)  # [hour]
PERIOD_DAYS = 30.0  # [day]
SOURCE_RADIUS_SOLAR = 1.0  # [solar radius]
SOURCE_MASS_SOLAR = 1.0  # [solar mass]
LENS_MASS_SOLAR = 0.6  # [solar mass]
IMPACT_PARAMETER = 0.2
PLANET_LENS_MASS_SOLAR = 9.55e-4  # [solar mass]
SYSTEM_LENS_MASSES = {
    r"Jupiter-mass planet ($1\,M_\mathrm{Jup}$)": PLANET_LENS_MASS_SOLAR,
    r"White dwarf ($0.6\,M_\odot$)": 0.6,
    r"Neutron star ($1.4\,M_\odot$)": 1.4,
    r"Black hole ($8\,M_\odot$)": 8.0,
}
EFFECT_CONFIGURATIONS = {
    "Fiducial limb darkening, $b=0.2$": ((0.4, 0.25), 0.2),
    "Uniform source, $b=0.2$": ((0.0, 0.0), 0.2),
    "Limb darkening, $b=0.8$": ((0.4, 0.25), 0.8),
}


def evaluate_systems() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Evaluate compact companions and controlled source-model comparisons."""

    common_arguments = {
        "period_days": PERIOD_DAYS,
        "source_radius_solar": SOURCE_RADIUS_SOLAR,
        "source_mass_solar": SOURCE_MASS_SOLAR,
    }
    system_fluxes = {
        label: ephesos.evaluate_self_lensing_model(
            TIME_HOURS / 24.0,
            lens_mass_solar=lens_mass_solar,
            impact_parameter=IMPACT_PARAMETER,
            **common_arguments,
        )
        for label, lens_mass_solar in SYSTEM_LENS_MASSES.items()
    }
    effect_fluxes = {
        label: ephesos.evaluate_self_lensing_model(
            TIME_HOURS / 24.0,
            lens_mass_solar=LENS_MASS_SOLAR,
            limb_darkening_coefficients=limb_darkening_coefficients,
            impact_parameter=impact_parameter,
            **common_arguments,
        )
        for label, (limb_darkening_coefficients, impact_parameter) in (
            EFFECT_CONFIGURATIONS.items()
        )
    }
    return TIME_HOURS, system_fluxes, effect_fluxes


def run_example(
    output_path: Path,
    typefileplot: str = "png",
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], Path]:
    """Plot deterministic system and finite-source effect comparisons."""

    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite self-lensing figure {output_path}")
    time_hours, system_fluxes, effect_fluxes = evaluate_systems()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.2, 9.6), layout="constrained")
    panels = figure.subfigures(3, 1)
    axes = [panel.subplots() for panel in panels]
    for axis in axes[1:]:
        axis.sharex(axes[0])
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    colors = ("#16697A", "#C73E1D", "#2D7D46")
    planet_label, planet_flux = next(iter(system_fluxes.items()))
    for color, (label, relative_flux) in zip(
        colors, tuple(system_fluxes.items())[1:]
    ):
        axes[0].plot(
            time_hours,
            1e6 * (relative_flux - 1.0),
            color=color,
            linewidth=2.0,
            label=label,
        )
    planet_excess_ppm = 1e6 * (planet_flux - 1.0)
    axes[1].plot(
        time_hours,
        planet_excess_ppm,
        color="#B06A00",
        linewidth=2.0,
        label=planet_label,
    )
    planet_peak_ppm = float(np.max(planet_excess_ppm))
    axes[1].set_ylim(-0.05 * planet_peak_ppm, 1.1 * planet_peak_ppm)
    for color, (label, relative_flux) in zip(colors, effect_fluxes.items()):
        axes[2].plot(
            time_hours,
            1e6 * (relative_flux - 1.0),
            color=color,
            linewidth=2.0,
            label=label,
        )

    axes[0].set_title("Compact-object mass controls self-lensing amplitude")
    axes[0].set_ylabel("Flux excess [ppm]")
    axes[1].set_title("A Jupiter-mass planet produces a sub-ppm lensing pulse")
    axes[1].set_ylabel("Flux excess [ppm]")
    axes[2].set_title(r"Source brightness and alignment reshape a $0.6\,M_\odot$ pulse")
    axes[2].set_xlabel("Time from conjunction [hour]")
    axes[2].set_ylabel("Flux excess [ppm]")
    for panel, axis in zip(panels, axes):
        axis.grid(False)
        handles, labels = axis.get_legend_handles_labels()
        panel.legend(handles, labels, loc="outside lower center", fancybox=True,
                     framealpha=1.0, fontsize=11.0)
        axis.tick_params(labelsize=11.0)
        axis.title.set_fontsize(11.0)
        axis.xaxis.label.set_size(11.0)
        axis.yaxis.label.set_size(11.0)
    axes[0].set_xlim(-4.0, 4.0)

    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if typefileplot == "png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return time_hours, system_fluxes, effect_fluxes, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).parent / "visuals" / f"self_lensing_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.png",
    )
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    run_example(arguments.output_path, typefileplot=arguments.typefileplot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())