#!/usr/bin/env python3
"""Forward-model a mapped planetary atmosphere through secondary eclipse."""

import argparse
from pathlib import Path

from tdpy.cli import add_plot_arguments

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import ephesos


PERIOD_DAYS = 1.0  # [day]
RADIUS_RATIO = 0.1
SUMMED_RADIUS_TO_SEMIMAJOR_AXIS = 0.2


def make_brightness_map(size: int = 201) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a toy projected brightness map with an off-center hotspot."""

    coordinates = np.linspace(-1.0, 1.0, size)
    map_x, map_y = np.meshgrid(coordinates, coordinates)
    companion_disk = map_x**2 + map_y**2 <= 1.0
    brightness_map = 0.002 + 0.035 * np.exp(
        -0.5 * (((map_x - 0.35) / 0.2) ** 2 + ((map_y + 0.1) / 0.3) ** 2)
    )
    brightness_map = np.where(companion_disk, brightness_map, 0.0)
    return brightness_map, map_x, map_y


def run_example(
    output_path: Path,
    typefileplot: str = "png",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    """Compare mapped and uniform planetary eclipse predictions."""

    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    orbital_phase = np.linspace(0.46, 0.54, 401)
    time_days = orbital_phase * PERIOD_DAYS  # [day]
    brightness_map, map_x, map_y = make_brightness_map()
    companion_disk = map_x**2 + map_y**2 <= 1.0
    mean_brightness = float(np.mean(brightness_map[companion_disk]))
    uniform_map = np.where(companion_disk, mean_brightness, 0.0)
    model_arguments = {
        "period_days": PERIOD_DAYS,
        "radius_ratio": RADIUS_RATIO,
        "summed_radius_to_semimajor_axis": SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
    }
    mapped_flux = ephesos.evaluate_eclipse_mapping_model(
        time_days,
        companion_brightness_map=brightness_map,
        **model_arguments,
    )
    uniform_flux = ephesos.evaluate_eclipse_mapping_model(
        time_days,
        companion_brightness_map=uniform_map,
        **model_arguments,
    )

    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(10.0, 5.6), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.6),
        height_ratios=(1.4, 0.8),
        wspace=0.58,
        hspace=0.22,
    )
    map_axis = figure.add_subplot(grid[:, 0])
    curve_axis = figure.add_subplot(grid[0, 1])
    residual_axis = figure.add_subplot(grid[1, 1], sharex=curve_axis)

    image = map_axis.imshow(
        np.ma.masked_where(~companion_disk, brightness_map),
        origin="lower",
        extent=(-1.0, 1.0, -1.0, 1.0),
        interpolation="nearest",
        cmap="inferno",
    )
    map_axis.add_patch(Circle((0.0, 0.0), 1.0, fill=False, color="white", linewidth=1.2))
    map_axis.set_aspect("equal")
    map_axis.set_xlim(-1.1, 1.1)
    map_axis.set_ylim(-1.1, 1.1)
    map_axis.set_xlabel(r"Projected $x/R_p$")
    map_axis.set_ylabel(r"Projected $y/R_p$")
    map_axis.set_title("Planetary brightness map")
    map_axis.grid(False)
    colorbar = figure.colorbar(image, ax=map_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Brightness ratio")

    curve_axis.plot(
        orbital_phase,
        1e6 * (mapped_flux - 1.0),
        color="#C73E1D",
        linewidth=2.0,
        label="Mapped planet",
    )
    curve_axis.plot(
        orbital_phase,
        1e6 * (uniform_flux - 1.0),
        color="#16697A",
        linewidth=1.8,
        linestyle="--",
        label="Uniform map, matched mean",
    )
    curve_axis.axvline(0.5, color="0.35", linewidth=0.8, linestyle=":")
    curve_axis.set_ylabel("Flux [ppm]")
    curve_axis.set_title("Secondary eclipse scans the bright region")
    curve_axis.legend(loc="upper right", fancybox=True, framealpha=1.0)
    curve_axis.grid(False)

    residual_axis.plot(
        orbital_phase,
        1e6 * (mapped_flux - uniform_flux),
        color="#2D7D46",
        linewidth=1.8,
    )
    residual_axis.axhline(0.0, color="black", linewidth=0.8)
    residual_axis.axvline(0.5, color="0.35", linewidth=0.8, linestyle=":")
    residual_axis.set_xlabel("Orbital phase")
    residual_axis.set_ylabel(r"$\Delta$ flux [ppm]")
    residual_axis.grid(False)
    curve_axis.tick_params(labelbottom=False)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.9)

    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if typefileplot == "png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return orbital_phase, mapped_flux, uniform_flux, brightness_map, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    output_path = Path(__file__).with_name("visuals") / "eclipse_mapping.png"
    run_example(output_path, typefileplot=arguments.typefileplot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
