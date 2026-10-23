#!/usr/bin/env python3
"""Compare equal-area circumplanetary-disk transit morphologies."""

from tdpy.verbosity import print

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
    if output_path.suffix.lower() not in (".png", ".pdf", ".html"):
        raise ValueError("output_path must have a .png, .pdf, or .html suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ("black", "#16697A", "#C73E1D", "#2D7D46")
    benchmark_flux = relative_flux["disk"]
    if output_path.suffix.lower() == ".html":
        from bokeh.layouts import column
        from bokeh.models import HoverTool
        from bokeh.plotting import figure as bokeh_figure
        from tdpy.plotting import save_bokeh_html

        tools = "pan,wheel_zoom,box_zoom,reset,save"
        flux_plot = bokeh_figure(
            title="Equal-area circumplanetary-disk transits",
            height=340,
            sizing_mode="stretch_width",
            tools=tools,
        )
        residual_plot = bokeh_figure(
            title="Disk-model departures from a spherical planet",
            height=260,
            sizing_mode="stretch_width",
            x_range=flux_plot.x_range,
            tools=tools,
        )
        for color, (occultor_type, label) in zip(colors, MORPHOLOGIES.items()):
            flux = relative_flux[occultor_type]
            flux_plot.line(time_hours, flux, color=color, line_width=2, legend_label=label)
            if occultor_type != "disk":
                residual_plot.line(
                    time_hours,
                    1e3 * (flux - benchmark_flux),
                    color=color,
                    line_width=2,
                    legend_label=label,
                )
        for plot, y_label, tooltips in (
            (flux_plot, "Relative flux", [("Flux", "$y")]),
            (residual_plot, "Departure [ppt]", [("Departure [ppt]", "$y")]),
        ):
            plot.grid.visible = False
            plot.xaxis.axis_label = "Time from mid-transit [hour]"
            plot.yaxis.axis_label = y_label
            plot.legend.location = "bottom_right"
            plot.add_tools(HoverTool(tooltips=[("Time [hour]", "$x"), *tooltips]))
        save_bokeh_html(column(flux_plot, residual_plot), output_path, title="Circumplanetary disks")
        return time_hours, relative_flux, output_path

    figure, (flux_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
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
        default=Path(__file__).parent / "visuals" / "circumplanetary_disk_morphologies.png",
    )
    parser.add_argument("--bokeh", action="store_true", help="write an interactive HTML version")
    arguments = parser.parse_args()
    output_path = arguments.output_path.with_suffix(".html") if arguments.bokeh else arguments.output_path
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
