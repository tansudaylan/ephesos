"""Visualization helpers for Ephesos model outputs."""

from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import corner
import numpy as np
from PIL import Image

from .geometry import OccultorType, projected_occultor_mask


PlotBackground = Literal["white", "dark"]
PlotFileType = Literal["png", "pdf"]
LightCurveAnimationMode = Literal["reveal", "trailing"]


def save_corner_figure(
    samples: np.ndarray,
    labels: list[str] | tuple[str, ...],
    output_path: Path | str,
    *,
    title: str,
    typefileplot: PlotFileType = "png",
    font_size: float = 11.0,  # [point]
) -> Path:
    """Write a corner plot summarizing a population of derived features."""

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] != len(labels):
        raise ValueError("samples must contain multiple rows and one column per label")
    if not np.isfinite(samples).all():
        raise ValueError("samples must contain only finite values")
    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")

    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = corner.corner(
        samples,
        labels=labels,
        bins=24,
        color="#16697A",
        plot_datapoints=True,
        plot_density=False,
        plot_contours=False,
        quantiles=(0.16, 0.5, 0.84),
        show_titles=True,
        label_kwargs={"fontsize": font_size},
        title_kwargs={"fontsize": font_size},
        hist_kwargs={"linewidth": 1.8},
        data_kwargs={"alpha": 0.45},
    )
    figure.suptitle(title, fontsize=font_size)
    for axis in figure.axes:
        axis.grid(False)
        axis.tick_params(labelsize=font_size)
    figure.set_facecolor("white")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300 if typefileplot == "png" else None, bbox_inches="tight")
    plt.close(figure)
    return output_path


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


def _select_animation_frame_indices(
    relative_flux: np.ndarray,
    max_frames: int,
) -> np.ndarray:
    """Sample the full sequence while concentrating frames on flux transitions."""

    if max_frames < 2:
        raise ValueError("max_frames must be at least two")
    frame_count = min(max_frames, relative_flux.size)
    if frame_count == relative_flux.size:
        return np.arange(frame_count)

    uniform_count = max(2, frame_count // 3)
    uniform_indices = np.linspace(0, relative_flux.size - 1, uniform_count, dtype=int)
    cumulative_variation = np.cumsum(np.abs(np.diff(relative_flux, prepend=relative_flux[0])))
    if cumulative_variation[-1] > 0.0:
        transition_count = frame_count - uniform_count
        variation_targets = np.linspace(
            0.0,
            cumulative_variation[-1],
            transition_count + 2,
        )[1:-1]
        transition_indices = np.searchsorted(cumulative_variation, variation_targets)
        selected_indices = np.unique(np.concatenate((uniform_indices, transition_indices)))
    else:
        selected_indices = uniform_indices

    # Fill duplicate quantiles by repeatedly bisecting the largest unsampled gaps.
    while selected_indices.size < frame_count:
        remaining_indices = np.setdiff1d(
            np.arange(relative_flux.size), selected_indices, assume_unique=True
        )
        nearest_distance = np.min(
            np.abs(remaining_indices[:, None] - selected_indices[None, :]), axis=1
        )
        selected_indices = np.sort(
            np.append(selected_indices, remaining_indices[np.argmax(nearest_distance)])
        )
    return selected_indices


def _estimate_transit_duration(time: np.ndarray, relative_flux: np.ndarray) -> float:
    """Estimate the median complete transit duration in the units of ``time``."""

    flux_deficit = np.clip(1.0 - relative_flux, 0.0, None)
    threshold = max(1e-8, 0.001 * np.max(flux_deficit))
    in_transit = flux_deficit > threshold
    boundaries = np.diff(in_transit.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(boundaries == 1)
    stops = np.flatnonzero(boundaries == -1) - 1
    complete = (starts > 0) & (stops < time.size - 1)
    durations = time[stops[complete]] - time[starts[complete]]
    if durations.size == 0 or not np.any(durations > 0.0):
        return 0.1 * (time[-1] - time[0])
    return float(np.median(durations[durations > 0.0]))


def _trailing_window_limits(
    time: np.ndarray,
    frame_index: int,
    history_duration: float,
) -> tuple[float, float]:
    """Return a fixed-width viewport ending at the current animation time."""

    current_time = float(time[frame_index])
    return current_time - history_duration, current_time


def _animation_frame_repetitions(
    projected_x: np.ndarray,
    projected_y: np.ndarray,
    occultor_radius: np.ndarray,
    *,
    transit_slowdown: int,
    ingress_egress_slowdown: int,
    simultaneous_transit_slowdown: int,
) -> np.ndarray:
    """Return state-aware frame repetitions from projected occultor geometry."""

    projected_distance = np.hypot(projected_x, projected_y)
    outer_contact = projected_distance <= 1.0 + occultor_radius[:, None]
    full_transit = projected_distance <= np.maximum(1.0 - occultor_radius, 0.0)[:, None]
    ingress_or_egress = outer_contact & ~full_transit
    active_count = np.count_nonzero(outer_contact, axis=0)

    repetitions = np.ones(projected_x.shape[1], dtype=int)
    repetitions[np.any(outer_contact, axis=0)] = transit_slowdown
    repetitions[np.any(ingress_or_egress, axis=0)] = ingress_egress_slowdown
    repetitions[active_count >= 2] = simultaneous_transit_slowdown
    return repetitions


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
    period: float | np.ndarray,
    radius_ratio: float | np.ndarray,
    summed_radius_to_semimajor_axis: float | np.ndarray,
    cosine_inclination: float | np.ndarray = 0.0,
    transit_epoch: float | np.ndarray = 0.0,
    limb_darkening_coefficients: tuple[float, float] = (0.4, 0.25),
    occultor_type: OccultorType = "disk",
    oblateness: float = 0.3,
    custom_occultor_mask: np.ndarray | None = None,
    custom_occultor_extent: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    comparison_relative_flux: np.ndarray | None = None,
    comparison_label: str = "Comparison model",
    comparison_models: dict[str, np.ndarray] | None = None,
    model_label: str = "Complete model",
    time_label: str = "Time [day]",
    max_frames: int = 72,
    frames_per_second: int = 24,  # [frame s^-1]
    animation_dpi: int = 100,  # [dot inch^-1]
    light_curve_mode: LightCurveAnimationMode = "reveal",
    history_duration: float | None = None,
    transit_slowdown: int = 1,
    ingress_egress_slowdown: int = 1,
    simultaneous_transit_slowdown: int = 1,
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
    period, radius_ratio, summed_radius_to_semimajor_axis, cosine_inclination, transit_epoch = (
        np.broadcast_arrays(
            np.atleast_1d(np.asarray(period, dtype=float)),
            np.atleast_1d(np.asarray(radius_ratio, dtype=float)),
            np.atleast_1d(np.asarray(summed_radius_to_semimajor_axis, dtype=float)),
            np.atleast_1d(np.asarray(cosine_inclination, dtype=float)),
            np.atleast_1d(np.asarray(transit_epoch, dtype=float)),
        )
    )
    if not all(
        np.isfinite(parameter).all()
        for parameter in (
            period,
            radius_ratio,
            summed_radius_to_semimajor_axis,
            cosine_inclination,
            transit_epoch,
        )
    ):
        raise ValueError("orbital parameters must contain only finite values")
    if np.any(period <= 0.0):
        raise ValueError("period must be positive")
    if np.any(radius_ratio <= 0.0):
        raise ValueError("radius_ratio must be positive")
    if np.any(summed_radius_to_semimajor_axis <= 0.0):
        raise ValueError("summed_radius_to_semimajor_axis must be positive")
    if np.any(np.abs(cosine_inclination) > 1.0):
        raise ValueError("cosine_inclination must be between -1 and 1")
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")
    if animation_dpi <= 0:
        raise ValueError("animation_dpi must be positive")
    if light_curve_mode not in ("reveal", "trailing"):
        raise ValueError("light_curve_mode must be 'reveal' or 'trailing'")
    if history_duration is not None and history_duration <= 0.0:
        raise ValueError("history_duration must be positive")
    if light_curve_mode == "trailing" and history_duration is None:
        history_duration = 3.0 * _estimate_transit_duration(time, relative_flux)
    slowdown_factors = (
        transit_slowdown,
        ingress_egress_slowdown,
        simultaneous_transit_slowdown,
    )
    if any(not isinstance(factor, int) or factor < 1 for factor in slowdown_factors):
        raise ValueError("animation slowdown factors must be positive integers")
    valid_occultor_types = (
        "disk",
        "oblate",
        "face_on_rings",
        "horizontal_rings",
        "vertical_rings",
        "custom",
    )
    if occultor_type not in valid_occultor_types:
        raise ValueError(f"occultor_type must be one of {valid_occultor_types}")
    if occultor_type == "custom" and custom_occultor_mask is None:
        raise ValueError("custom_occultor_mask is required for a custom occultor")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("output_path must have a .gif suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = _select_animation_frame_indices(relative_flux, max_frames)
    plotted_flux = np.concatenate((relative_flux, *validated_comparison_models.values()))
    flux_span = np.ptp(plotted_flux)
    flux_padding = max(0.08 * flux_span, 1e-4)
    colors = _plot_colors(typeplotback)

    # Calculate the circular sky-plane orbit in units of the stellar radius.
    semimajor_axis_stellar_radii = (1.0 + radius_ratio) / summed_radius_to_semimajor_axis
    orbital_phase = 2.0 * np.pi * (time[None, :] - transit_epoch[:, None]) / period[:, None]
    projected_x = semimajor_axis_stellar_radii[:, None] * np.sin(orbital_phase)
    projected_y = (
        semimajor_axis_stellar_radii[:, None] * cosine_inclination[:, None] * np.cos(orbital_phase)
    )
    if occultor_type == "custom":
        outer_radius_factor = max(abs(bound) for bound in custom_occultor_extent)
    else:
        outer_radius_factor = 1.75 if "rings" in occultor_type else 1.0
    frame_repetitions = _animation_frame_repetitions(
        projected_x,
        projected_y,
        outer_radius_factor * radius_ratio,
        transit_slowdown=transit_slowdown,
        ingress_egress_slowdown=ingress_egress_slowdown,
        simultaneous_transit_slowdown=simultaneous_transit_slowdown,
    )
    animation_frame_indices = np.repeat(frame_indices, frame_repetitions[frame_indices])

    # Render the same quadratic limb-darkened stellar surface assumed by the model.
    image_limit = max(1.25, 1.1 * 1.75 * np.max(radius_ratio))
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
    figure.suptitle(title, color=colors["foreground"], fontsize=font_size)
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
    image_axis.set_title("Image", fontsize=font_size)
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
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    def update(frame_index: int):
        occulted = np.zeros_like(stellar_disk)
        for companion_index in range(period.size):
            distance_x = image_x - projected_x[companion_index, frame_index]
            distance_y = image_y - projected_y[companion_index, frame_index]
            occulted |= projected_occultor_mask(
                distance_x,
                distance_y,
                radius_ratio[companion_index],
                occultor_type,
                oblateness=oblateness,
                custom_mask=custom_occultor_mask,
                custom_mask_extent=custom_occultor_extent,
            )
        current_brightness = np.where(occulted, 0.0, stellar_brightness)
        model_image.set_data(current_brightness)
        stop = frame_index + 1
        if light_curve_mode == "trailing":
            window_start, window_end = _trailing_window_limits(
                time,
                frame_index,
                history_duration,
            )
            start = np.searchsorted(time, window_start)
            curve_axis.set_xlim(window_start, window_end)
        else:
            start = 0
        model_line.set_data(time[start:stop], relative_flux[start:stop])
        current_point.set_data([time[frame_index]], [relative_flux[frame_index]])
        return model_image, model_line, current_point

    animation = FuncAnimation(
        figure,
        update,
        frames=animation_frame_indices,
        blit=light_curve_mode == "reveal",
    )
    print(f"Writing to {output_path}...")
    animation.save(
        output_path,
        writer=PillowWriter(fps=frames_per_second),
        dpi=animation_dpi,
    )
    plt.close(figure)
    return output_path
