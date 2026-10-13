#!/usr/bin/env python3
"""Compare planet-only and illustrative leading- and trailing-moon transits."""

from tdpy.verbosity import print

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import ephesos


TIME_HOURS = np.linspace(-6.0, 6.0, 241)  # [hour]
PLANET_PERIOD_DAYS = 4.0  # [day]
PLANET_RADIUS_RATIO = 0.09
SUMMED_RADIUS_TO_SEMIMAJOR_AXIS = 0.1
MOON_RADIUS_RATIO = 0.025
MOON_SEPARATION_PLANET_RADII = 5.0
MOON_PERIOD_DAYS = 0.5  # [day]
COSINE_INCLINATION = 0.015
MOON_PHASES = {
    "Leading moon": np.pi / 2.0,  # [rad]
    "Trailing moon": -np.pi / 2.0,  # [rad]
}


def evaluate_models() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate a planet-only curve and two circular moon-orbit phases."""

    time_days = TIME_HOURS / 24.0  # [day]
    common_arguments = {
        "planet_period_days": PLANET_PERIOD_DAYS,
        "planet_radius_ratio": PLANET_RADIUS_RATIO,
        "summed_radius_to_semimajor_axis": SUMMED_RADIUS_TO_SEMIMAJOR_AXIS,
        "moon_separation_planet_radii": MOON_SEPARATION_PLANET_RADII,
        "moon_period_days": MOON_PERIOD_DAYS,
        "cosine_inclination": COSINE_INCLINATION,
        "grid_size": 201,
    }
    relative_flux = {
        "Planet only": ephesos.evaluate_exomoon_transit_model(
            time_days,
            moon_radius_ratio=0.0,
            **common_arguments,
        )
    }
    relative_flux.update(
        {
            label: ephesos.evaluate_exomoon_transit_model(
                time_days,
                moon_radius_ratio=MOON_RADIUS_RATIO,
                moon_orbital_phase_radians=phase,
                **common_arguments,
            )
            for label, phase in MOON_PHASES.items()
        }
    )
    return TIME_HOURS, relative_flux


def run_example(
    output_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], Path]:
    """Plot deterministic toy exomoon transits and their planet-only residuals."""

    time_hours, relative_flux = evaluate_models()
    output_path = Path(output_path)
    if output_path.suffix.lower() not in (".png", ".pdf", ".html"):
        raise ValueError("output_path must have a .png, .pdf, or .html suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".html":
        from bokeh.layouts import column
        from bokeh.models import HoverTool
        from bokeh.plotting import figure as bokeh_figure
        from tdpy.plotting import save_bokeh_html

        colors = {
            "Planet only": "#20252B",
            "Leading moon": "#16697A",
            "Trailing moon": "#C73E1D",
        }
        shared_tools = "pan,wheel_zoom,box_zoom,reset,save"
        flux_plot = bokeh_figure(
            title="Planet and exomoon transit signals",
            width=720,
            height=340,
            sizing_mode="stretch_width",
            tools=shared_tools,
        )
        residual_plot = bokeh_figure(
            title="Moon signal relative to the planet-only model",
            width=720,
            height=280,
            sizing_mode="stretch_width",
            x_range=flux_plot.x_range,
            tools=shared_tools,
        )
        planet_only_flux = relative_flux["Planet only"]
        for label, flux in relative_flux.items():
            flux_plot.line(
                time_hours,
                flux,
                color=colors[label],
                line_width=2.2,
                legend_label=label,
            )
            if label != "Planet only":
                residual_plot.line(
                    time_hours,
                    1e3 * (flux - planet_only_flux),
                    color=colors[label],
                    line_width=2.2,
                    legend_label=label,
                )
        for plot in (flux_plot, residual_plot):
            plot.grid.visible = False
            plot.xaxis.axis_label = "Time from planetary transit [hour]"
            plot.add_tools(
                HoverTool(tooltips=[("Time [hour]", "$x"), ("Signal", "$y")])
            )
            plot.legend.location = "bottom_right"
        flux_plot.yaxis.axis_label = "Relative flux"
        residual_plot.yaxis.axis_label = "Moon signal [ppt]"
        plot = column(flux_plot, residual_plot, sizing_mode="stretch_width")
        save_bokeh_html(plot, output_path, title="Exomoon transit models")
        return time_hours, relative_flux, output_path

    figure, (flux_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    colors = {
        "Planet only": "black",
        "Leading moon": "#16697A",
        "Trailing moon": "#C73E1D",
    }
    planet_only_flux = relative_flux["Planet only"]
    for label, flux in relative_flux.items():
        flux_axis.plot(time_hours, flux, color=colors[label], label=label)
        if label != "Planet only":
            residual_axis.plot(
                time_hours,
                1e3 * (flux - planet_only_flux),
                color=colors[label],
                label=label,
            )
    flux_axis.set_ylabel("Relative flux")
    flux_axis.set_title("Planet and exomoon transit signals")
    residual_axis.set_xlabel("Time from planetary transit [hour]")
    residual_axis.set_ylabel("Moon signal [ppt]")
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
        default=Path(__file__).parent / "visuals" / "exomoon_transits.png",
    )
    parser.add_argument("--bokeh", action="store_true", help="write an interactive HTML version")
    arguments = parser.parse_args()
    output_path = arguments.output_path.with_suffix(".html") if arguments.bokeh else arguments.output_path
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())