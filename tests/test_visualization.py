from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest
from PIL import Image

import ephesos
from ephesos.visualization import (
    OrbitingBody,
    _animation_frame_repetitions,
    _estimate_transit_duration,
    _select_animation_frame_indices,
    _trailing_window_limits,
    save_orbital_geometry_figure,
)
from ephesos.geometry import orbital_euler_rotation_stages


def sample_light_curve() -> tuple[np.ndarray, np.ndarray]:
    """Return a compact deterministic transit-shaped test curve."""

    time_hours = np.linspace(-2.0, 2.0, 21)  # [hour]
    relative_flux = 1.0 - 0.01 * np.exp(-0.5 * time_hours**2)
    return time_hours, relative_flux


def test_model_background_selection_preserves_flux_and_caller_style():
    import matplotlib.pyplot as plt
    from tdpy.plotting import plot_style

    time = np.linspace(-0.2, 0.2, 21)  # [day]
    original_time = time.copy()
    outputs = []
    with plot_style('white'):
        original_settings = dict(plt.rcParams)
        for typeplotback in ('white', 'dark'):
            outputs.append(ephesos.eval_modl(
                time, 'PlanetarySystem', pericomp=np.array([3.2]), epocmtracomp=np.array([0.0]),
                rsmacomp=np.array([0.085]), cosicomp=np.array([0.0]), rratcomp=np.array([0.085]),
                typelmdk='quad', booldiag=False, typeverb=0, typeplotback=typeplotback,
            ))
            assert dict(plt.rcParams) == original_settings
    np.testing.assert_array_equal(outputs[0]['rflx'], outputs[1]['rflx'])
    np.testing.assert_array_equal(time, original_time)


def test_orbital_euler_rotation_stages_follow_z_x_z_order() -> None:
    stages = orbital_euler_rotation_stages(
        np.array((0.0, 1.0, 0.0)),
        np.pi / 2.0,
        np.pi / 2.0,
        np.pi / 2.0,
    )

    assert len(stages) == 4
    np.testing.assert_allclose(stages[0], (0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(stages[1], (-1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(stages[2], (-1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(stages[3], (0.0, -1.0, 0.0), atol=1e-12)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 3), (0, 3)])
def test_orbital_euler_rotation_stages_preserve_batch_shape_and_intermediates(shape):
    position = np.arange(np.prod(shape), dtype=float).reshape(shape)
    original = position.copy()
    angles = (0.31, -0.72, 1.17)  # [rad]
    cosine = np.cos(angles)
    sine = np.sin(angles)
    matrices = (
        np.array([[cosine[0], -sine[0], 0.0], [sine[0], cosine[0], 0.0], [0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, cosine[1], -sine[1]], [0.0, sine[1], cosine[1]]]),
        np.array([[cosine[2], -sine[2], 0.0], [sine[2], cosine[2], 0.0], [0.0, 0.0, 1.0]]),
    )
    stages = orbital_euler_rotation_stages(position, *angles)
    expected = position.copy()
    np.testing.assert_array_equal(stages[0], original)
    assert not np.shares_memory(stages[0], position)
    for stage, matrix in zip(stages[1:], matrices):
        expected = np.einsum('ij,...j->...i', matrix, expected)
        assert stage.shape == shape
        np.testing.assert_allclose(stage, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(position, original)


@pytest.mark.parametrize("position, angle", [([1.0, 2.0], 0.1), ([np.nan, 0.0, 0.0], 0.1), ([1.0, 0.0, 0.0], np.inf)])
def test_orbital_euler_rotation_stages_retain_invalid_input_rejection(position, angle):
    with pytest.raises(ValueError, match="finite"):
        orbital_euler_rotation_stages(position, angle, 0.2, 0.3)


@pytest.mark.parametrize("verbosity", [None, "0", "1"])
def test_save_orbital_geometry_figure_includes_finite_observer_view(
    tmp_path: Path,
    capsys,
    monkeypatch,
    verbosity,
) -> None:
    if verbosity is None:
        monkeypatch.delenv("TDPY_VERBOSITY", raising=False)
    else:
        monkeypatch.setenv("TDPY_VERBOSITY", verbosity)
    orbiting_bodies = (
        OrbitingBody("Planet b", 18.0, 0.12, 25.0, 38.0, 15.0, 65.0),
        OrbitingBody("Planet c", 34.0, 0.08, 130.0, 62.0, 95.0, 210.0),
    )
    output_path = save_orbital_geometry_figure(
        orbiting_bodies,
        tmp_path / "orbital_geometry.png",
        viewing_angles_deg=((35.0, 25.0), (125.0, 35.0)),
        observer_positions=((7.0, -4.0, 3.0),),
        field_of_view_deg=130.0,
    )

    assert output_path.is_file() and output_path.stat().st_size > 10_000
    output = capsys.readouterr().out
    if verbosity == "1":
        assert f"Writing to {output_path}..." in output
    else:
        assert output == ""
    with Image.open(output_path) as image:
        assert image.width > image.height
        assert image.width > 2_000

    with pytest.raises(ValueError, match="outside the star"):
        save_orbital_geometry_figure(
            orbiting_bodies,
            tmp_path / "invalid.png",
            observer_positions=((0.0, 0.0, 0.0),),
        )


def test_animation_sampling_concentrates_frames_on_ingress_and_egress() -> None:
    relative_flux = np.ones(401)
    relative_flux[100:121] = np.linspace(1.0, 0.99, 21)
    relative_flux[121:280] = 0.99
    relative_flux[280:301] = np.linspace(0.99, 1.0, 21)

    frame_indices = _select_animation_frame_indices(relative_flux, max_frames=72)
    transition_frames = np.count_nonzero(
        ((frame_indices >= 100) & (frame_indices <= 120))
        | ((frame_indices >= 280) & (frame_indices <= 300))
    )

    assert frame_indices.size == 72
    assert transition_frames >= 36
    assert frame_indices[0] == 0
    assert frame_indices[-1] == relative_flux.size - 1


def test_animation_history_defaults_to_three_transit_durations() -> None:
    time_hours = np.linspace(0.0, 10.0, 101)  # [hour]
    relative_flux = np.ones_like(time_hours)
    relative_flux[10:13] = 0.99
    relative_flux[40:43] = 0.99

    transit_duration = _estimate_transit_duration(time_hours, relative_flux)

    assert 3.0 * transit_duration == pytest.approx(0.6)


def test_trailing_window_ends_at_current_time() -> None:
    time_hours = np.linspace(0.0, 10.0, 101)  # [hour]

    limits = _trailing_window_limits(time_hours, frame_index=60, history_duration=2.5)

    assert limits == pytest.approx((3.5, 6.0))


def test_animation_slowdown_prioritizes_transit_states() -> None:
    projected_x = np.array(((2.0, 1.05, 0.0), (2.0, 2.0, 0.0)))
    projected_y = np.zeros_like(projected_x)

    repetitions = _animation_frame_repetitions(
        projected_x,
        projected_y,
        np.array((0.1, 0.1)),
        transit_slowdown=2,
        ingress_egress_slowdown=4,
        simultaneous_transit_slowdown=6,
    )

    assert np.array_equal(repetitions, (1, 4, 6))


@pytest.mark.parametrize("typefileplot", ("png", "pdf", "html"))
@pytest.mark.parametrize("verbosity", [None, "0", "1"])
def test_save_light_curve_figure_supports_required_formats(
    tmp_path: Path,
    capsys,
    typefileplot: str,
    monkeypatch,
    verbosity,
) -> None:
    if verbosity is None:
        monkeypatch.delenv("TDPY_VERBOSITY", raising=False)
    else:
        monkeypatch.setenv("TDPY_VERBOSITY", verbosity)
    if typefileplot == "html":
        pytest.importorskip("bokeh")
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
    if typefileplot == "html":
        assert "Bokeh" in result.read_text(encoding="utf-8")
    output = capsys.readouterr().out
    if verbosity == "1":
        assert f"Writing to {result}..." in output
    else:
        assert output == ""


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
            period=24.0,  # [hour]
            radius_ratio=0.1,
            summed_radius_to_semimajor_axis=0.1,
            typeplotback="gray",
        )


@pytest.mark.parametrize(
    "occultor_type",
    ("face_on_rings", "horizontal_rings", "vertical_rings"),
)
def test_save_light_curve_animation_has_model_and_comparison_panels(
    tmp_path: Path,
    occultor_type: str,
) -> None:
    time_hours, relative_flux = sample_light_curve()
    output_path = tmp_path / f"{occultor_type}.gif"

    ephesos.save_light_curve_animation(
        time_hours,
        relative_flux,
        output_path,
        title="Deterministic transit",
        period=24.0,  # [hour]
        radius_ratio=0.1,
        summed_radius_to_semimajor_axis=0.1,
        occultor_type=occultor_type,
        comparison_models={
            "Spherical planet": relative_flux + 0.002,
            "Oblate planet": relative_flux + 0.001,
        },
        time_label="Time from mid-transit [hour]",
        max_frames=5,
    )

    with Image.open(output_path) as animation:
        animation.seek(animation.n_frames // 2)
        frame = np.asarray(animation.convert("RGB"))
        assert animation.n_frames > 1
        assert frame.shape[1] > 2 * frame.shape[0]
        assert np.std(frame[:, : frame.shape[1] // 2]) > 20.0


def test_save_light_curve_animation_renders_custom_occultor(tmp_path: Path) -> None:
    time_hours, relative_flux = sample_light_curve()
    coordinates = np.linspace(-1.5, 1.5, 101)
    image_x, image_y = np.meshgrid(coordinates, coordinates)
    custom_mask = (image_x**2 + image_y**2 <= 1.0) | (
        (image_x > 0.8) & (np.abs(image_y) < 0.2)
    )
    output_path = tmp_path / "custom.gif"

    ephesos.save_light_curve_animation(
        time_hours,
        relative_flux,
        output_path,
        title="Custom transiter",
        period=24.0,  # [hour]
        radius_ratio=0.1,
        summed_radius_to_semimajor_axis=0.1,
        occultor_type="custom",
        custom_occultor_mask=custom_mask,
        custom_occultor_extent=(-1.5, 1.5, -1.5, 1.5),
        max_frames=5,
    )

    with Image.open(output_path) as animation:
        assert animation.n_frames > 1


def test_save_light_curve_animation_rejects_mismatched_comparison(tmp_path: Path) -> None:
    time_hours, relative_flux = sample_light_curve()

    with pytest.raises(ValueError, match="matching one-dimensional arrays"):
        ephesos.save_light_curve_animation(
            time_hours,
            relative_flux,
            tmp_path / "invalid.gif",
            title="Deterministic transit",
            period=24.0,  # [hour]
            radius_ratio=0.1,
            summed_radius_to_semimajor_axis=0.1,
            comparison_relative_flux=relative_flux[:-1],
        )


def test_save_light_curve_animation_rejects_invalid_history_options(tmp_path: Path) -> None:
    time_hours, relative_flux = sample_light_curve()

    with pytest.raises(ValueError, match="light_curve_mode"):
        ephesos.save_light_curve_animation(
            time_hours,
            relative_flux,
            tmp_path / "invalid.gif",
            title="Deterministic transit",
            period=24.0,  # [hour]
            radius_ratio=0.1,
            summed_radius_to_semimajor_axis=0.1,
            light_curve_mode="window",
        )
    with pytest.raises(ValueError, match="history_duration"):
        ephesos.save_light_curve_animation(
            time_hours,
            relative_flux,
            tmp_path / "invalid.gif",
            title="Deterministic transit",
            period=24.0,  # [hour]
            radius_ratio=0.1,
            summed_radius_to_semimajor_axis=0.1,
            light_curve_mode="trailing",
            history_duration=0.0,
        )
    with pytest.raises(ValueError, match="slowdown factors"):
        ephesos.save_light_curve_animation(
            time_hours,
            relative_flux,
            tmp_path / "invalid.gif",
            title="Deterministic transit",
            period=24.0,  # [hour]
            radius_ratio=0.1,
            summed_radius_to_semimajor_axis=0.1,
            transit_slowdown=1.5,
        )


def test_save_frame_animation_combines_frames(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
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
