"""Visualization helpers for Ephesos model outputs."""

from tdpy.verbosity import print

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import corner
import numpy as np
from PIL import Image
from tdpy.exoplanet import quadratic_limb_darkened_stellar_grid
from tdpy.plotting import save_bokeh_html, save_figure

from .geometry import (
    OccultorType,
    orbital_euler_rotation_stages,
    projected_occultor_mask,
)


PlotBackground = Literal["white", "dark"]
PlotFileType = Literal["png", "pdf", "html"]
LightCurveAnimationMode = Literal["reveal", "trailing"]


@dataclass(frozen=True)
class OrbitingBody:
    """Orbital elements and drawing radius for one orbiting body."""

    label: str
    semimajor_axis: float
    eccentricity: float
    argument_of_periapsis_deg: float
    inclination_deg: float
    longitude_of_ascending_node_deg: float
    true_anomaly_deg: float


def _orbital_position_stages(
    body: OrbitingBody,
    true_anomaly_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radius = (
        body.semimajor_axis
        * (1.0 - body.eccentricity**2)
        / (1.0 + body.eccentricity * np.cos(true_anomaly_rad))
    )
    ephesos_anomaly = true_anomaly_rad + np.pi / 2.0
    position = np.stack(
        (
            radius * np.cos(ephesos_anomaly),
            radius * np.sin(ephesos_anomaly),
            np.zeros_like(radius),
        ),
        axis=-1,
    )
    return orbital_euler_rotation_stages(
        position,
        np.deg2rad(body.argument_of_periapsis_deg),
        np.deg2rad(body.inclination_deg),
        np.deg2rad(body.longitude_of_ascending_node_deg),
    )


def _project_from_observer(
    positions: np.ndarray,
    observer_position: np.ndarray,
    focal_length: float,
    near_distance: float,
) -> np.ndarray:
    forward = -observer_position / np.linalg.norm(observer_position)
    reference_up = np.array((0.0, 0.0, 1.0))
    right = np.cross(forward, reference_up)
    if np.linalg.norm(right) < 1e-10:
        right = np.cross(forward, np.array((0.0, 1.0, 0.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    relative_position = positions - observer_position
    depth = relative_position @ forward
    projected = np.full((*positions.shape[:-1], 2), np.nan)
    visible = depth > near_distance
    projected[..., 0][visible] = (
        focal_length * (relative_position @ right)[visible] / depth[visible]
    )
    projected[..., 1][visible] = focal_length * (relative_position @ up)[visible] / depth[visible]
    return projected


def _project_from_direction(positions: np.ndarray, viewing_direction: np.ndarray) -> np.ndarray:
    forward = -viewing_direction / np.linalg.norm(viewing_direction)
    reference_up = np.array((0.0, 0.0, 1.0))
    right = np.cross(forward, reference_up)
    if np.linalg.norm(right) < 1e-10:
        right = np.cross(forward, np.array((0.0, 1.0, 0.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack((positions @ right, positions @ up), axis=-1)


def save_orbital_geometry_figure(
    orbiting_bodies: Sequence[OrbitingBody],
    output_path: Path | str,
    *,
    star_radius: float = 1.0,
    length_unit: str = "stellar radii",
    viewing_angles_deg: Sequence[tuple[float, float]] = ((35.0, 25.0),),
    observer_positions: Sequence[tuple[float, float, float]] = (),
    field_of_view_deg: float = 95.0,
    orbit_samples: int = 361,
    typefileplot: PlotFileType = "png",
    font_size: float = 10.0,  # [point]
) -> Path:
    """Plot Z-X-Z orbital rotation stages and perspective views of a system.

    Distances and observer positions use ``length_unit``. Euler angles and
    viewing angles are in degrees. Angle-based views are orthographic. Finite
    observers look toward the star in perspective and must lie outside it.
    Orbital positions use Ephesos's 90-degree phase offset before the
    periapsis, inclination, and ascending-node rotations.
    """

    orbiting_bodies = tuple(orbiting_bodies)
    if not orbiting_bodies:
        raise ValueError("orbiting_bodies must contain at least one body")
    if not all(isinstance(body, OrbitingBody) for body in orbiting_bodies):
        raise ValueError("orbiting_bodies must contain OrbitingBody instances")
    labels = [body.label for body in orbiting_bodies]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("body labels must be nonempty and unique")
    body_parameters = np.asarray(
        [
            (
                body.semimajor_axis,
                body.eccentricity,
                body.argument_of_periapsis_deg,
                body.inclination_deg,
                body.longitude_of_ascending_node_deg,
                body.true_anomaly_deg,
            )
            for body in orbiting_bodies
        ],
        dtype=float,
    )
    if not np.isfinite(body_parameters).all():
        raise ValueError("orbital elements and body radii must be finite")
    if any(body.semimajor_axis <= 0.0 for body in orbiting_bodies):
        raise ValueError("semimajor axes must be positive")
    if any(not 0.0 <= body.eccentricity < 1.0 for body in orbiting_bodies):
        raise ValueError("eccentricities must be between zero and one")
    if not np.isfinite(star_radius) or star_radius <= 0.0:
        raise ValueError("star_radius must be finite and positive")
    if not length_unit:
        raise ValueError("length_unit must be nonempty")
    if typefileplot not in ("png", "pdf"):
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    if not np.isfinite(field_of_view_deg) or not 1.0 <= field_of_view_deg < 179.0:
        raise ValueError("field_of_view_deg must be between 1 and 179 degrees")
    if not isinstance(orbit_samples, (int, np.integer)) or orbit_samples < 64:
        raise ValueError("orbit_samples must be an integer of at least 64")

    viewing_angles = np.asarray(viewing_angles_deg, dtype=float)
    if viewing_angles.size == 0:
        viewing_angles = np.empty((0, 2))
    if (
        viewing_angles.ndim != 2
        or viewing_angles.shape[1] != 2
        or not np.isfinite(viewing_angles).all()
        or np.any(np.abs(viewing_angles[:, 1]) > 90.0)
    ):
        raise ValueError("viewing_angles_deg must contain finite (azimuth, elevation) pairs")

    observers = np.asarray(observer_positions, dtype=float)
    if observers.size == 0:
        observers = np.empty((0, 3))
    if observers.ndim != 2 or observers.shape[1] != 3 or not np.isfinite(observers).all():
        raise ValueError("observer_positions must contain finite three-dimensional positions")
    if np.any(np.linalg.norm(observers, axis=1) <= star_radius):
        raise ValueError("finite observers must lie outside the star")

    anomalies = np.linspace(0.0, 2.0 * np.pi, orbit_samples)
    current_anomalies = np.deg2rad([body.true_anomaly_deg for body in orbiting_bodies])
    orbit_stages = [_orbital_position_stages(body, anomalies) for body in orbiting_bodies]
    body_stages = [
        _orbital_position_stages(body, np.array((anomaly,)))
        for body, anomaly in zip(orbiting_bodies, current_anomalies)
    ]
    maximum_orbit_radius = max(
        body.semimajor_axis * (1.0 + body.eccentricity) for body in orbiting_bodies
    )
    map_radius = (
        max(
            maximum_orbit_radius,
            float(np.max(np.linalg.norm(observers[:, :2], axis=1))) if observers.size else 0.0,
            star_radius,
        )
        * 1.08
    )
    focal_length = 1.0 / np.tan(np.deg2rad(field_of_view_deg) / 2.0)
    near_distance = max(1e-8, 1e-5 * maximum_orbit_radius)

    angle_views = []
    for azimuth_deg, elevation_deg in viewing_angles:
        azimuth_rad, elevation_rad = np.deg2rad((azimuth_deg, elevation_deg))
        direction = np.array(
            (
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad),
            )
        )
        angle_views.append(
            (
                f"View az={azimuth_deg:g} deg, el={elevation_deg:g} deg",
                direction,
            )
        )
    finite_views = [
        (
            "Observer at ("
            + ", ".join(f"{coordinate:g}" for coordinate in observer)
            + f") {length_unit}",
            observer,
        )
        for observer in observers
    ]

    panel_count = 4 + len(angle_views) + len(finite_views)
    column_count = min(4, panel_count)
    row_count = int(np.ceil(panel_count / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(3.5 * column_count, 3.25 * row_count),
        squeeze=False,
        facecolor="white",
    )
    flat_axes = axes.ravel()
    colors = [plt.get_cmap("tab10")(index % 10) for index in range(len(orbiting_bodies))]
    stage_titles = (
        "Orbital plane before rotations",
        "After Rz(omega)",
        "After Rx(inclination)",
        "After Rz(ascending node)",
    )

    for stage_index, axis in enumerate(flat_axes[:4]):
        for body_index, (body, color) in enumerate(zip(orbiting_bodies, colors)):
            orbit = orbit_stages[body_index][stage_index]
            position = body_stages[body_index][stage_index][0]
            axis.plot(orbit[:, 0], orbit[:, 1], color=color, linewidth=1.6)
            axis.scatter(position[0], position[1], color=[color], s=28, zorder=4)
        axis.add_patch(
            Circle(
                (0.0, 0.0),
                star_radius,
                facecolor="#E69F00",
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )
        )
        if stage_index == 3:
            axis.scatter(
                observers[:, 0],
                observers[:, 1],
                marker="x",
                color="black",
                s=42,
                zorder=6,
            )
        axis.set_title(stage_titles[stage_index], fontsize=font_size)
        axis.set_xlabel(f"X [{length_unit}]", fontsize=font_size)
        axis.set_ylabel(f"Y [{length_unit}]", fontsize=font_size)
        axis.set_xlim(-map_radius, map_radius)
        axis.set_ylim(-map_radius, map_radius)
        axis.set_aspect("equal", adjustable="box")
        axis.axhline(0.0, color="0.82", linewidth=0.6, zorder=0)
        axis.axvline(0.0, color="0.82", linewidth=0.6, zorder=0)
        axis.grid(False)
        axis.tick_params(labelsize=font_size)

    for view_index, (view_title, direction) in enumerate(angle_views, start=4):
        axis = flat_axes[view_index]
        for body_index, (body, color) in enumerate(zip(orbiting_bodies, colors)):
            orbit = _project_from_direction(orbit_stages[body_index][3], direction)
            position = _project_from_direction(body_stages[body_index][3], direction)[0]
            axis.plot(orbit[:, 0], orbit[:, 1], color=color, linewidth=1.6)
            if np.isfinite(position).all():
                axis.scatter(position[0], position[1], color=[color], s=28, zorder=4)
        axis.scatter(0.0, 0.0, color="#E69F00", edgecolor="black", s=90, zorder=5)
        axis.set_title(view_title, fontsize=font_size)
        axis.set_xlabel(f"Projected X [{length_unit}]", fontsize=font_size)
        axis.set_ylabel(f"Projected Y [{length_unit}]", fontsize=font_size)
        axis.set_xlim(-map_radius, map_radius)
        axis.set_ylim(-map_radius, map_radius)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(False)
        axis.tick_params(labelsize=font_size)

    finite_view_start = 4 + len(angle_views)
    for view_index, (view_title, observer) in enumerate(finite_views, start=finite_view_start):
        axis = flat_axes[view_index]
        for body_index, (body, color) in enumerate(zip(orbiting_bodies, colors)):
            orbit = _project_from_observer(
                orbit_stages[body_index][3], observer, focal_length, near_distance
            )
            position = _project_from_observer(
                body_stages[body_index][3], observer, focal_length, near_distance
            )[0]
            axis.plot(orbit[:, 0], orbit[:, 1], color=color, linewidth=1.6)
            if np.isfinite(position).all():
                axis.scatter(position[0], position[1], color=[color], s=28, zorder=4)
        star_position = _project_from_observer(
            np.array(((0.0, 0.0, 0.0),)), observer, focal_length, near_distance
        )[0]
        if np.isfinite(star_position).all():
            axis.scatter(
                star_position[0],
                star_position[1],
                color="#E69F00",
                edgecolor="black",
                s=90,
                zorder=5,
            )
        axis.set_title(view_title, fontsize=font_size)
        axis.set_xlabel("Image-plane x", fontsize=font_size)
        axis.set_ylabel("Image-plane y", fontsize=font_size)
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(False)
        axis.tick_params(labelsize=font_size)

    for axis in flat_axes[panel_count:]:
        axis.set_axis_off()

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="#E69F00",
            markeredgecolor="black",
            markersize=8,
            label="Star",
        )
    ]
    legend_handles.extend(
        Line2D([], [], color=color, marker="o", label=body.label)
        for body, color in zip(orbiting_bodies, colors)
    )
    if observers.size:
        legend_handles.append(
            Line2D([], [], color="black", marker="x", linestyle="none", label="Finite observer")
        )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 5),
        fancybox=True,
        framealpha=1.0,
        fontsize=font_size,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return Path(
        save_figure(
            figure,
            output_path,
            typefileplot=typefileplot,
            close_figure=True,
        )
    )


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
    return Path(
        save_figure(
            figure,
            output_path,
            typefileplot=typefileplot,
            close_figure=True,
        )
    )


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
    if typefileplot not in ("png", "pdf", "html"):
        raise ValueError("typefileplot must be 'png', 'pdf', or 'html'")

    output_path = Path(output_path).with_suffix(f".{typefileplot}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = _plot_colors(typeplotback)
    if typefileplot == "html":
        from bokeh.models import HoverTool
        from bokeh.plotting import figure as bokeh_figure

        plot = bokeh_figure(
            title=title,
            width=720,
            height=420,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        plot.line(
            time,
            relative_flux,
            color=colors["model"],
            line_width=2.3,
            legend_label=model_label,
        )
        plot.line(
            (time[0], time[-1]),
            (1.0, 1.0),
            color=colors["foreground"],
            line_width=1.0,
            line_dash="dashed",
            legend_label="Unocculted flux",
        )
        plot.add_tools(
            HoverTool(tooltips=[("Time", "$x"), ("Relative flux", "$y")])
        )
        plot.xaxis.axis_label = time_label
        plot.yaxis.axis_label = "Relative flux"
        plot.grid.visible = False
        plot.background_fill_color = colors["background"]
        plot.border_fill_color = colors["background"]
        plot.title.text_color = colors["foreground"]
        plot.xaxis.axis_label_text_color = colors["foreground"]
        plot.yaxis.axis_label_text_color = colors["foreground"]
        plot.legend.location = "bottom_right"
        plot.legend.background_fill_color = colors["background"]
        plot.legend.border_line_color = colors["foreground"]
        return save_bokeh_html(plot, output_path, title=title)

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
    image_x, image_y, radial_distance, stellar_brightness = quadratic_limb_darkened_stellar_grid(
        181,
        limb_darkening_coefficients,
        image_limit=image_limit,
    )
    stellar_disk = radial_distance <= 1.0

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
        occulted = np.zeros_like(stellar_disk, dtype=float)
        for companion_index in range(period.size):
            distance_x = image_x - projected_x[companion_index, frame_index]
            distance_y = image_y - projected_y[companion_index, frame_index]
            opacity = projected_occultor_mask(
                distance_x,
                distance_y,
                radius_ratio[companion_index],
                occultor_type,
                oblateness=oblateness,
                custom_mask=custom_occultor_mask,
                custom_mask_extent=custom_occultor_extent,
            )
            occulted = 1.0 - (1.0 - occulted) * (1.0 - opacity)
        current_brightness = (1.0 - occulted) * stellar_brightness
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
