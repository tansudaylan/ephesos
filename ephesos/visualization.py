"""Visualization helpers for Ephesos model outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def save_light_curve_animation(
    time: np.ndarray,
    relative_flux: np.ndarray,
    output_path: Path | str,
    *,
    title: str,
    time_label: str = "Time [day]",
    max_frames: int = 24,
    frames_per_second: int = 12,
) -> Path:
    """Write a compact GIF that progressively reveals a model light curve."""

    time = np.asarray(time, dtype=float)
    relative_flux = np.asarray(relative_flux, dtype=float)
    if time.ndim != 1 or relative_flux.shape != time.shape or time.size < 2:
        raise ValueError("time and relative_flux must be matching one-dimensional arrays")
    if not np.isfinite(time).all() or not np.isfinite(relative_flux).all():
        raise ValueError("animation inputs must contain only finite values")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("output_path must have a .gif suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = np.unique(
        np.linspace(0, time.size - 1, min(max_frames, time.size), dtype=int)
    )
    flux_span = np.ptp(relative_flux)
    flux_padding = max(0.08 * flux_span, 1e-4)

    figure, axis = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    axis.set_facecolor("white")
    axis.plot(time, relative_flux, color="0.8", linewidth=1.2)
    model_line, = axis.plot([], [], color="#16697A", linewidth=2.3)
    current_point, = axis.plot([], [], "o", color="#C73E1D", markersize=6)
    axis.set_xlim(time[0], time[-1])
    axis.set_ylim(relative_flux.min() - flux_padding, relative_flux.max() + flux_padding)
    axis.set_xlabel(time_label)
    axis.set_ylabel("Relative flux")
    axis.set_title(title)
    axis.grid(False)
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