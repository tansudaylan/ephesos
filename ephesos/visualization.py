"""Visualization helpers for Ephesos model outputs."""

from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image


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
    time_label: str = "Time [day]",
    max_frames: int = 24,
    frames_per_second: int = 12,  # [frame s^-1]
    typeplotback: PlotBackground = "white",
    font_size: float = 11.0,  # [point]
) -> Path:
    """Write a compact GIF that progressively reveals a model light curve."""

    time, relative_flux = _validate_light_curve(time, relative_flux)

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("output_path must have a .gif suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = np.unique(np.linspace(0, time.size - 1, min(max_frames, time.size), dtype=int))
    flux_span = np.ptp(relative_flux)
    flux_padding = max(0.08 * flux_span, 1e-4)
    colors = _plot_colors(typeplotback)

    figure, axis = plt.subplots(
        figsize=(7.2, 4.2),
        facecolor=colors["background"],
    )
    axis.plot(
        time,
        relative_flux,
        color=colors["context"],
        linewidth=1.2,
        label="Complete model",
    )
    (model_line,) = axis.plot(
        [],
        [],
        color=colors["model"],
        linewidth=2.3,
        label="Revealed model",
    )
    (current_point,) = axis.plot(
        [],
        [],
        "o",
        color=colors["current"],
        markersize=6,
        label="Current time",
    )
    axis.set_xlim(time[0], time[-1])
    axis.set_ylim(relative_flux.min() - flux_padding, relative_flux.max() + flux_padding)
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

    def update(frame_index: int):
        stop = frame_index + 1
        model_line.set_data(time[:stop], relative_flux[:stop])
        current_point.set_data([time[frame_index]], [relative_flux[frame_index]])
        return model_line, current_point

    animation = FuncAnimation(figure, update, frames=frame_indices, blit=True)
    print(f"Writing to {output_path}...")
    animation.save(output_path, writer=PillowWriter(fps=frames_per_second))
    plt.close(figure)
    return output_path
