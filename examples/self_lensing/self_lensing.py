#!/usr/bin/env python3
"""Compare toy self-lensing systems and finite-source effects."""

import argparse
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
SYSTEM_LENS_MASSES = {
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
    time_hours, system_fluxes, effect_fluxes = evaluate_systems()
    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), sharex=True)
    colors = ("#16697A", "#C73E1D", "#2D7D46")
    for color, (label, relative_flux) in zip(colors, system_fluxes.items()):
        axes[0].plot(
            time_hours,
            1e6 * (relative_flux - 1.0),
            color=color,
            linewidth=2.0,
            label=label,
        )
    for color, (label, relative_flux) in zip(colors, effect_fluxes.items()):
        axes[1].plot(
            time_hours,
            1e6 * (relative_flux - 1.0),
            color=color,
            linewidth=2.0,
            label=label,
        )

    axes[0].set_title("Compact-object mass controls self-lensing amplitude")
    axes[0].set_ylabel("Flux excess [ppm]")
    axes[1].set_title(r"Source brightness and alignment reshape a $0.6\,M_\odot$ pulse")
    axes[1].set_xlabel("Time from conjunction [hour]")
    axes[1].set_ylabel("Flux excess [ppm]")
    for axis in axes:
        axis.grid(False)
        axis.legend(loc="upper right", fancybox=True, framealpha=1.0, fontsize=11.0)
        axis.tick_params(labelsize=11.0)
        axis.title.set_fontsize(11.0)
        axis.xaxis.label.set_size(11.0)
        axis.yaxis.label.set_size(11.0)
    figure.tight_layout()

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
        default=Path(__file__).with_name("self_lensing.png"),
    )
    parser.add_argument("--typefileplot", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()
    run_example(arguments.output_path, typefileplot=arguments.typefileplot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())