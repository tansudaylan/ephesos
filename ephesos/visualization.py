"""Visualization helpers for Ephesos model outputs."""

from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image

from .geometry import OccultorType, projected_occultor_mask


PlotBackground = Literal["white", "dark"]
PlotFileType = Literal["png", "pdf"]


def _validate_light_curve(
    time: np.ndarray, relative_flux: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, matching one-dimensional light-curve arrays."""

    time = np.asarray(time, dtype=float)
    relative_flux = np.asarray(relative_flux, dtype=float)
    if time.ndim != 1 or relative_flux.shape != time.shape or time.size < 2:
        raise ValueError("time and relative_flux must be matching one-dimensional arrays")
    if not np.isfinite(time).all() or not np.isfinite(relative_flux).all():
        raise ValueError("light-curve inputs must contain only finite values")
    return time, relative_flux


def _plot_colors(typeplotback: PlotBackground) -> dict[str, str]:
    """Return an accessible light-curve palette for the requested background."""

    if typeplotback == "white":
        return {
            "background": "white",
            "foreground": "black",
            "context": "0.75",
            "model": "#16697A",
            "current": "#C73E1D",
        }
    if typeplotback == "dark":
        return {
            "background": "#111111",
            "foreground": "white",
            "context": "0.45",
            "model": "#58C4DD",
            "current": "#FFC857",
        }
    raise ValueError("typeplotback must be 'white' or 'dark'")


def _style_light_curve_axis(
    axis,
    *,
    title: str,
    time_label: str,
    colors: dict[str, str],
    font_size: float,
) -> None:
    """Apply the shared Ephesos light-curve figure style."""

    axis.set_facecolor(colors["background"])
    axis.set_xlabel(time_label, color=colors["foreground"], fontsize=font_size)
    axis.set_ylabel("Relative flux", color=colors["foreground"], fontsize=font_size)
    axis.set_title(title, color=colors["foreground"], fontsize=font_size)
    axis.tick_params(colors=colors["foreground"], labelsize=font_size)
    for spine in axis.spines.values():
        spine.set_color(colors["foreground"])
    axis.grid(False)


def save_light_curve_figure(
    time: np.ndarray,
    relative_flux: np.ndarray,
    output_path: Path | str,
    *,
    title: str,
    time_label: str = "Time [day]",
    model_label: str = "Ephesos model",
    typefileplot: PlotFileType = "png",
    typeplotback: PlotBackground = "white",
    font_size: float = 11.0,  # [point]
) -> Path:
    """Write a focused, publication-ready light-curve figure."""

    time, relative_flux = _validate_light_curve(time, relative_flux)
    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")

    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = _plot_colors(typeplotback)
    figure, axis = plt.subplots(
        figsize=(7.2, 4.2),
        facecolor=colors["background"],
    )
    axis.plot(
        time,
        relative_flux,
        color=colors["model"],
        linewidth=2.3,
        label=model_label,
    )
    axis.axhline(
        1.0,
        color=colors["foreground"],
        linewidth=1.0,
        linestyle="--",
        label="Unocculted flux",
    )
    _style_light_curve_axis(
        axis,
        title=title,
        time_label=time_label,
        colors=colors,
        font_size=font_size,
    )
    legend = axis.legend(
        loc="lower right",
        fancybox=True,
        framealpha=1.0,
        fontsize=font_size,
    )
    legend.get_frame().set_facecolor(colors["background"])
    legend.get_frame().set_edgecolor(colors["foreground"])
    for text in legend.get_texts():
        text.set_color(colors["foreground"])
    figure.tight_layout()

    print(f"Writing to {output_path}...")
    save_options = {
        "dpi": 300 if typefileplot == "png" else None,
        "facecolor": colors["background"],
    }
    figure.savefig(output_path, bbox_inches="tight", **save_options)
    plt.close(figure)
    return output_path


def save_frame_animation(
    frame_paths: list[Path | str],
    output_path: Path | str,
    *,
    frame_duration_ms: int = 50,  # [ms]
) -> Path:
    """Combine ordered raster frames into a looping GIF without shell tools."""

    frame_paths = [Path(path) for path in frame_paths]
    if len(frame_paths) < 2:
        raise ValueError("at least two frame paths are required")
    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("output_path must have a .gif suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for frame_path in frame_paths:
        print(f"Reading from {frame_path}...")
        with Image.open(frame_path) as frame:
            frames.append(frame.convert("RGBA"))

    try:
        print(f"Writing to {output_path}...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )
    finally:
        for frame in frames:
            frame.close()
    return output_path


def save_light_curve_animation(
    time: np.ndarray,
    relative_flux: np.ndarray,
    output_path: Path | str,
    *,
    title: str,
    period: float,
    radius_ratio: float,
    summed_radius_to_semimajor_axis: float,
    cosine_inclination: float = 0.0,
    limb_darkening_coefficients: tuple[float, float] = (0.4, 0.25),
    occultor_type: OccultorType = "disk",
    oblateness: float = 0.3,
    comparison_relative_flux: np.ndarray | None = None,
    comparison_label: str = "Comparison model",
    comparison_models: dict[str, np.ndarray] | None = None,
    model_label: str = "Complete model",
    time_label: str = "Time [day]",
    max_frames: int = 24,
    frames_per_second: int = 12,  # [frame s^-1]
    typeplotback: PlotBackground = "white",
    font_size: float = 11.0,  # [point]
) -> Path:
    """Write a synchronized sky-plane model and light-curve GIF."""

    time, relative_flux = _validate_light_curve(time, relative_flux)
    validated_comparison_models = {}
    if comparison_relative_flux is not None:
        _, comparison_relative_flux = _validate_light_curve(time, comparison_relative_flux)
        validated_comparison_models[comparison_label] = comparison_relative_flux
    if comparison_models is not None:
        for label, comparison_flux in comparison_models.items():
            _, validated_comparison_models[label] = _validate_light_curve(time, comparison_flux)
    if period <= 0.0:
        raise ValueError("period must be positive")
    if radius_ratio <= 0.0:
        raise ValueError("radius_ratio must be positive")
    if summed_radius_to_semimajor_axis <= 0.0:
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if not -1.0 <= cosine_inclination <= 1.0:
        raise ValueError("cosine_inclination must be between -1 and 1")
    valid_occultor_types = (
        "disk",
        "oblate",
        "face_on_rings",
        "horizontal_rings",
        "vertical_rings",
    )
    if occultor_type not in valid_occultor_types:
        raise ValueError(f"occultor_type must be one of {valid_occultor_types}")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("output_path must have a .gif suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = np.unique(np.linspace(0, time.size - 1, min(max_frames, time.size), dtype=int))
    plotted_flux = np.concatenate((relative_flux, *validated_comparison_models.values()))
    flux_span = np.ptp(plotted_flux)
    flux_padding = max(0.08 * flux_span, 1e-4)
    colors = _plot_colors(typeplotback)

    # Calculate the circular sky-plane orbit in units of the stellar radius.
    semimajor_axis_stellar_radii = (1.0 + radius_ratio) / summed_radius_to_semimajor_axis
    orbital_phase = 2.0 * np.pi * time / period
    projected_x = semimajor_axis_stellar_radii * np.sin(orbital_phase)
    projected_y = semimajor_axis_stellar_radii * cosine_inclination * np.cos(orbital_phase)

    # Render the same quadratic limb-darkened stellar surface assumed by the model.
    image_limit = max(1.25, 1.1 * 1.75 * radius_ratio)
    image_coordinates = np.linspace(-image_limit, image_limit, 181)
    image_x, image_y = np.meshgrid(image_coordinates, image_coordinates)
    radial_distance = np.sqrt(image_x**2 + image_y**2)
    stellar_disk = radial_distance <= 1.0
    cosine_emission_angle = np.sqrt(np.clip(1.0 - radial_distance**2, 0.0, 1.0))
    linear_coefficient, quadratic_coefficient = limb_darkening_coefficients
    stellar_brightness = np.zeros_like(radial_distance)
    stellar_brightness[stellar_disk] = (
        1.0
        - linear_coefficient * (1.0 - cosine_emission_angle[stellar_disk])
        - quadratic_coefficient * (1.0 - cosine_emission_angle[stellar_disk]) ** 2
    )

    figure, (image_axis, curve_axis) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.2),
        facecolor=colors["background"],
        gridspec_kw={"width_ratios": (1.0, 1.45)},
    )
    model_image = image_axis.imshow(
        stellar_brightness,
        origin="lower",
        extent=(-image_limit, image_limit, -image_limit, image_limit),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    image_axis.set_aspect("equal")
    image_axis.set_xlim(-image_limit, image_limit)
    image_axis.set_ylim(-image_limit, image_limit)
    image_axis.set_xlabel(r"Sky position [$R_\star$]", fontsize=font_size)
    image_axis.set_ylabel(r"Sky position [$R_\star$]", fontsize=font_size)
    image_axis.set_title(title, fontsize=font_size)
    image_axis.tick_params(colors=colors["foreground"], labelsize=font_size)
    image_axis.xaxis.label.set_color(colors["foreground"])
    image_axis.yaxis.label.set_color(colors["foreground"])
    image_axis.title.set_color(colors["foreground"])
    image_axis.set_facecolor(colors["background"])
    image_axis.grid(False)
    for spine in image_axis.spines.values():
        spine.set_color(colors["foreground"])

    curve_axis.plot(
        time,
        relative_flux,
        color=colors["context"],
        linewidth=1.2,
        label=model_label,
    )
    comparison_colors = (colors["foreground"], "#2D7D46", "#9C5A18")
    comparison_linestyles = ("--", ":", "-.")
    for index, (label, comparison_flux) in enumerate(validated_comparison_models.items()):
        curve_axis.plot(
            time,
            comparison_flux,
            color=comparison_colors[index % len(comparison_colors)],
            linewidth=1.5,
            linestyle=comparison_linestyles[index % len(comparison_linestyles)],
            label=label,
        )
    (model_line,) = curve_axis.plot(
        [],
        [],
        color=colors["model"],
        linewidth=2.3,
    )
    (current_point,) = curve_axis.plot(
        [],
        [],
        "o",
        color=colors["current"],
        markersize=6,
    )
    curve_axis.set_xlim(time[0], time[-1])
    curve_axis.set_ylim(
        plotted_flux.min() - flux_padding,
        plotted_flux.max() + flux_padding,
    )
    _style_light_curve_axis(
        curve_axis,
        title="Light Curve",
        time_label=time_label,
        colors=colors,
        font_size=font_size,
    )
    legend = curve_axis.legend(
        loc="lower right",
        fancybox=True,
        framealpha=1.0,
        fontsize=font_size,
    )
    legend.get_frame().set_facecolor(colors["background"])
    legend.get_frame().set_edgecolor(colors["foreground"])
    for text in legend.get_texts():
        text.set_color(colors["foreground"])
    figure.tight_layout()

    def update(frame_index: int):
        distance_x = image_x - projected_x[frame_index]
        distance_y = image_y - projected_y[frame_index]
        occulted = projected_occultor_mask(
            distance_x,
            distance_y,
            radius_ratio,
            occultor_type,
            oblateness=oblateness,
        )
        current_brightness = np.where(occulted, 0.0, stellar_brightness)
        model_image.set_data(current_brightness)
        stop = frame_index + 1
        model_line.set_data(time[:stop], relative_flux[:stop])
        current_point.set_data([time[frame_index]], [relative_flux[frame_index]])
        return model_image, model_line, current_point

    animation = FuncAnimation(figure, update, frames=frame_indices, blit=True)
    print(f"Writing to {output_path}...")
    animation.save(output_path, writer=PillowWriter(fps=frames_per_second))
    plt.close(figure)
    return output_path
