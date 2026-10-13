#!/usr/bin/env python3
"""Predict a light curve from explicitly specified evolving spots and flares."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ephesos import evaluate_spot_flare_model
from tdpy.cli import parse_plot_arguments
from tdpy.verbosity import print


def run_example(output_path):
    time_days = np.linspace(0.0, 30.0, 601)  # [day]
    components = evaluate_spot_flare_model(
        time_days, rotation_period_days=10.0, inclination_degrees=75.0,
        relative_shear=0.2, spot_latitudes_degrees=[15.0, -30.0],
        spot_longitudes_degrees=[0.0, 120.0], spot_radius_ratios=[0.1, 0.07],
        spot_peak_times_days=[10.0, 20.0], spot_widths_days=[5.0, 7.0],
        flare_peak_times_days=[12.0, 23.0], flare_amplitudes=[0.02, 0.01],
        flare_rise_times_days=[0.04, 0.06], flare_decay_times_days=[0.3, 0.5],
        return_components=True,
    )
    figure, axis = plt.subplots(figsize=(8.0, 4.5), facecolor="white")
    axis.plot(time_days, components["flux"], color="black", linewidth=1.8, label="Total model")
    axis.plot(time_days, 1.0 + np.sum(components["evolved_spot_flux"] - 1.0, axis=0),
              color="#007A6F", linestyle="--", label="Evolving spots")
    axis.plot(time_days, 1.0 + components["flare_flux"], color="#A51417", alpha=0.7, label="Flares")
    axis.set(xlabel="Elapsed time [day]", ylabel="Relative flux",
             title="Simulated evolving spots and flares", xlim=(0.0, 30.0))
    axis.grid(False)
    axis.legend(frameon=True, fancybox=True, framealpha=1.0)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"time_days": time_days, **components}


def main():
    arguments = parse_plot_arguments(description=__doc__)
    run_example(Path(__file__).with_name(f"spot_flare.{arguments.typefileplot}"))


if __name__ == "__main__":
    main()