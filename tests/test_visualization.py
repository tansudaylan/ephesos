from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest
from PIL import Image

import ephesos


def sample_light_curve() -> tuple[np.ndarray, np.ndarray]:
    """Return a compact deterministic transit-shaped test curve."""

    time_hours = np.linspace(-2.0, 2.0, 21)  # [hour]
    relative_flux = 1.0 - 0.01 * np.exp(-0.5 * time_hours**2)
    return time_hours, relative_flux


@pytest.mark.parametrize("typefileplot", ("png", "pdf"))
def test_save_light_curve_figure_supports_required_formats(
    tmp_path: Path,
    capsys,
    typefileplot: str,
) -> None:
    time_hours, relative_flux = sample_light_curve()
    output_path = tmp_path / "light_curve"

    result = ephesos.save_light_curve_figure(
        time_hours,
        relative_flux,
        output_path,
        title="Deterministic transit",
        time_label="Time from mid-transit [hour]",
        typefileplot=typefileplot,
    )

    assert result == output_path.with_suffix(f".{typefileplot}")
    assert result.is_file()
    assert result.stat().st_size > 1_000
    assert f"Writing to {result}..." in capsys.readouterr().out


def test_save_light_curve_figure_uses_white_background_by_default(tmp_path: Path) -> None:
    time_hours, relative_flux = sample_light_curve()

    output_path = ephesos.save_light_curve_figure(
        time_hours,
        relative_flux,
        tmp_path / "white_background",
        title="Deterministic transit",
        time_label="Time from mid-transit [hour]",
    )

    image = mpimg.imread(output_path)
    assert np.all(image[0, 0, :3] > 0.98)


def test_save_light_curve_figure_supports_dark_background(tmp_path: Path) -> None:
    time_hours, relative_flux = sample_light_curve()

    output_path = ephesos.save_light_curve_figure(
        time_hours,
        relative_flux,
        tmp_path / "dark_background",
        title="Deterministic transit",
        time_label="Time from mid-transit [hour]",
        typeplotback="dark",
    )

    image = mpimg.imread(output_path)
    assert np.all(image[0, 0, :3] < 0.1)


def test_save_light_curve_figure_rejects_invalid_options(tmp_path: Path) -> None:
    time_hours, relative_flux = sample_light_curve()

    with pytest.raises(ValueError, match="typefileplot"):
        ephesos.save_light_curve_figure(
            time_hours,
            relative_flux,
            tmp_path / "invalid",
            title="Deterministic transit",
            typefileplot="jpeg",
        )

    with pytest.raises(ValueError, match="typeplotback"):
        ephesos.save_light_curve_animation(
            time_hours,
            relative_flux,
            tmp_path / "invalid.gif",
            title="Deterministic transit",
            typeplotback="gray",
        )


def test_save_frame_animation_combines_frames(tmp_path: Path, capsys) -> None:
    frame_paths = [tmp_path / f"frame_{index}.png" for index in range(2)]
    for index, frame_path in enumerate(frame_paths):
        print(f"Writing to {frame_path}...")
        Image.new("RGB", (20, 20), color=(index * 255, 0, 0)).save(frame_path)

    output_path = tmp_path / "animation.gif"
    result = ephesos.save_frame_animation(frame_paths, output_path)

    with Image.open(result) as animation:
        assert animation.n_frames == 2
    output = capsys.readouterr().out
    assert all(f"Reading from {path}..." in output for path in frame_paths)
    assert f"Writing to {output_path}..." in output
