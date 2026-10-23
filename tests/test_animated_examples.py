import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


EXAMPLE_DIRECTORY = Path(__file__).parents[1] / "examples"


def test_only_supported_standalone_examples_are_at_root():
    standalone_examples = {
        path.name
        for path in EXAMPLE_DIRECTORY.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    assert standalone_examples <= {"differential_rotation.py", "spot_flare.py"}


def load_example(name):
    path = EXAMPLE_DIRECTORY / name / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_notebook(name):
    path = EXAMPLE_DIRECTORY / name / f"{name}.ipynb"
    print(f"Reading from {path}...")
    notebook = json.loads(path.read_text())
    code = "\n".join(
        "\n".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return path, notebook, code


def test_exomoon_example_writes_interactive_bokeh_plot(tmp_path):
    pytest.importorskip("bokeh")
    output_path = tmp_path / "exomoon.html"

    time_hours, relative_flux, result_path = load_example("exomoon").run_example(
        output_path
    )

    assert result_path == output_path
    assert time_hours.ndim == 1
    assert len(relative_flux) == 3
    html = output_path.read_text(encoding="utf-8")
    assert "Bokeh" in html
    assert "Planet and exomoon transit signals" in html


def test_planet_disk_example_writes_interactive_bokeh_plot(tmp_path):
    pytest.importorskip("bokeh")
    output_path = tmp_path / "planet_disk.html"

    time_hours, relative_flux, result_path = load_example("PlanetsWithDisks").run_example(
        output_path
    )

    assert result_path == output_path
    assert time_hours.ndim == 1
    assert set(relative_flux) == {"disk", "face_on_rings", "horizontal_rings", "vertical_rings"}
    html = output_path.read_text(encoding="utf-8")
    assert "Bokeh" in html
    assert "Equal-area circumplanetary-disk transits" in html
    assert "Disk-model departures from a spherical planet" in html


@pytest.mark.parametrize(
    "name",
    ("arbitrary_occultor", "run_WhiteDwarf", "run_WASP43"),
)
def test_single_system_examples_write_animations(name, tmp_path):
    output_path = tmp_path / f"{name}.gif"

    time, relative_flux = load_example(name).run_example(output_path)

    assert time.shape == relative_flux.shape
    assert output_path.is_file()
    assert output_path.stat().st_size > 1_000
    with Image.open(output_path) as image:
        assert image.n_frames > 1


def test_compact_multiplanet_example_writes_unique_history_modes(tmp_path):
    time, relative_flux, output_paths = load_example("compact_multiplanet").run_example(
        tmp_path / "compact_multiplanet.gif",
        max_frames=5,
    )

    assert time.shape == relative_flux.shape
    assert {path.name for path in output_paths} == {
        "compact_multiplanet_reveal.gif",
        "compact_multiplanet_trailing.gif",
    }
    for output_path in output_paths:
        with Image.open(output_path) as image:
            assert image.n_frames > 1


def test_population_example_writes_derived_feature_corner_plot(tmp_path):
    example = load_example("run_population")
    assert example.DEFAULT_SAMPLE_SIZE >= 2048
    assert example.DEFAULT_SAMPLE_SIZE % len(example.OCCULTOR_TYPES) == 0
    features, output_path = example.run_example(
        tmp_path / "population_features.png",
        sample_size=24,
    )

    assert features.shape == (24, 8)
    assert np.isfinite(features).all()
    assert np.all(np.ptp(features, axis=0) > 0.0)
    assert all(
        "departure" in label.lower() for label in example.FEATURE_LABELS[4:]
    )
    assert {"oblate", "face_on_rings", "horizontal_rings", "vertical_rings"} == set(
        example.OCCULTOR_TYPES
    )
    assert output_path.is_file() and output_path.stat().st_size > 10_000
    with Image.open(output_path) as image:
        assert image.n_frames == 1
        assert image.width > 1_000
        assert image.height > 1_000


def test_self_lensing_example_writes_brightening_figure(tmp_path):
    example = load_example("self_lensing")

    time_hours, system_fluxes, effect_fluxes, output_path = example.run_example(
        tmp_path / "self_lensing.png"
    )

    all_fluxes = (*system_fluxes.values(), *effect_fluxes.values())
    assert len(system_fluxes) == 4
    assert len(effect_fluxes) == 3
    assert all(relative_flux.shape == time_hours.shape for relative_flux in all_fluxes)
    assert all(np.isfinite(relative_flux).all() for relative_flux in all_fluxes)
    assert all(np.argmax(relative_flux) == relative_flux.size // 2 for relative_flux in all_fluxes)
    system_peaks = [np.max(relative_flux) for relative_flux in system_fluxes.values()]
    assert np.all(np.diff(system_peaks[1:]) > 0.0)
    assert 0.01 < 1e6 * (system_peaks[0] - 1.0) < 0.1
    effect_peaks = [np.max(relative_flux) for relative_flux in effect_fluxes.values()]
    assert effect_peaks[0] > effect_peaks[1]
    assert effect_peaks[0] > effect_peaks[2]
    assert 500.0 < 1e6 * (system_peaks[1] - 1.0) < 1_000.0
    assert output_path.is_file() and output_path.stat().st_size > 10_000
    with Image.open(output_path) as image:
        assert image.n_frames == 1
        assert image.height > image.width


def test_orbital_geometry_example_writes_multiview_figure(tmp_path):
    example = load_example("orbital_geometry")

    output_path = example.run_example(tmp_path / "orbital_geometry.png")

    assert output_path.is_file() and output_path.stat().st_size > 10_000
    with Image.open(output_path) as image:
        assert image.n_frames == 1
        assert image.width > image.height


def test_photodynamical_ttv_example_plots_ephesos_predictions(tmp_path, monkeypatch):
    example = load_example("photodynamical_ttv")
    predictions = [
        (np.arange(4), np.array((1.0, 11.0, 21.1, 31.0))),
        (np.arange(3), np.array((2.0, 17.1, 32.0))),
    ]
    calls = []

    def predict(planet_parameters, stellar_mass, start_time, step, end_time):
        calls.append((planet_parameters.copy(), stellar_mass, start_time, step, end_time))
        return predictions

    monkeypatch.setattr(example.ephesos, "evaluate_nbody_transit_times", predict)
    residuals, output_path = example.run_example(tmp_path / "photodynamical_ttv.png")

    assert len(residuals) == len(predictions) == 2
    assert all(np.isfinite(values).all() for _, values in residuals)
    assert calls[0][0].shape == (2, 7)
    assert output_path.is_file() and output_path.stat().st_size > 10_000
    with Image.open(output_path) as image:
        assert image.n_frames == 1
        assert image.width > image.height


def test_emitting_companion_example_resolves_secondary_eclipse(tmp_path):
    example = load_example("emitting_companion")

    orbital_phase, relative_flux, output_path = example.run_example(
        tmp_path / "emitting_companion.png"
    )

    eclipse_index = np.argmin(np.abs(orbital_phase - 0.5))
    quadrature_index = np.argmin(np.abs(orbital_phase - 0.25))
    quadrature_flux = []
    for flux in relative_flux.values():
        assert flux.shape == orbital_phase.shape
        assert np.isfinite(flux).all()
        assert flux[eclipse_index] == pytest.approx(1.0, abs=1e-10)
        quadrature_flux.append(flux[quadrature_index])
    assert np.all(np.diff(quadrature_flux) > 0.0)
    assert output_path.is_file() and output_path.stat().st_size > 10_000
    with Image.open(output_path) as image:
        assert image.n_frames == 1
        assert image.width > image.height


def test_known_self_lensers_compare_kepler_data_with_models(tmp_path):
    example = load_example("known_self_lensers")

    comparisons, output_path = example.run_example(
        tmp_path / "known_self_lensers.png"
    )

    assert set(comparisons) == {"KOI-3278", "KIC 8145411"}
    for time_hours, relative_flux, relative_flux_error, model_flux in comparisons.values():
        assert time_hours.shape == relative_flux.shape == relative_flux_error.shape
        assert model_flux.shape == time_hours.shape
        assert time_hours.size >= 30
        assert all(
            np.isfinite(values).all()
            for values in (time_hours, relative_flux, relative_flux_error, model_flux)
        )
        assert np.all(relative_flux_error > 0.0)
        assert np.max(model_flux) > 1.0005
        assert abs(time_hours[np.argmax(relative_flux)]) < 5.0
        assert np.mean(relative_flux[np.abs(time_hours) < 4.0]) > np.mean(
            relative_flux[np.abs(time_hours) > 8.0]
        )
    assert output_path.is_file() and output_path.stat().st_size > 10_000
    with Image.open(output_path) as image:
        assert image.n_frames == 1
        assert image.width > image.height


def test_planets_with_disks_compares_equal_area_morphologies(tmp_path):
    example = load_example("PlanetsWithDisks")
    time_hours, relative_flux, output_path = example.run_example(
        tmp_path / "PlanetsWithDisks.png"
    )

    assert set(relative_flux) == set(example.MORPHOLOGIES)
    assert all(flux.shape == time_hours.shape for flux in relative_flux.values())
    assert all(np.isfinite(flux).all() for flux in relative_flux.values())
    assert all(
        not np.allclose(relative_flux["disk"], relative_flux[occultor_type])
        for occultor_type in example.MORPHOLOGIES
        if occultor_type != "disk"
    )
    assert output_path.is_file() and output_path.stat().st_size > 10_000
    with Image.open(output_path) as image:
        assert image.n_frames == 1
        assert image.width > image.height


def test_simultaneous_transits_combine_two_individual_models(tmp_path):
    example = load_example("transits_simultaneous")
    time_hours, combined_flux, individual_flux, _ = example.evaluate_system()
    midpoint = time_hours.size // 2

    assert combined_flux.shape == time_hours.shape
    assert all(flux.shape == time_hours.shape for flux in individual_flux)
    assert np.isfinite(combined_flux).all()
    assert combined_flux[midpoint] < min(flux[midpoint] for flux in individual_flux)

    output_path = tmp_path / "transits_simultaneous.gif"
    returned_time, returned_flux = example.run_example(output_path, max_frames=5)
    assert np.array_equal(returned_time, time_hours)
    assert np.array_equal(returned_flux, combined_flux)
    with Image.open(output_path) as animation:
        assert animation.n_frames > 1


def test_equal_area_occultor_shapes_produce_finite_distinct_models():
    example = load_example("arbitrary_occultor")
    ringed_curves = []

    for orientation in example.RING_CONFIGURATIONS:
        _, ringed_flux, spherical_flux, oblate_flux = example.evaluate_occultor_models(orientation)
        assert np.isfinite(ringed_flux).all()
        assert np.isfinite(spherical_flux).all()
        assert np.isfinite(oblate_flux).all()
        assert not np.allclose(spherical_flux, oblate_flux)
        ringed_curves.append(ringed_flux)

    assert not np.allclose(ringed_curves[0], ringed_curves[1])
    assert not np.allclose(ringed_curves[1], ringed_curves[2])


def test_astromusers_logo_transit_compares_with_circular_core(tmp_path):
    example = load_example("astromusers_logo_occultor")
    time, logo_flux, circle_flux, logo_mask, _ = example.evaluate_occultor_models()

    assert time.shape == logo_flux.shape == circle_flux.shape
    assert logo_mask.ndim == 2 and logo_mask.any()
    assert np.isfinite(logo_flux).all()
    assert np.min(logo_flux) < np.min(circle_flux)
    assert not np.allclose(logo_flux, circle_flux)

    output_path = tmp_path / "astromusers_logo_occultor.gif"
    example.run_example(output_path, max_frames=5)
    with Image.open(output_path) as animation:
        assert animation.n_frames > 1


@pytest.mark.parametrize(
    "name",
    (
        "run_WASP43",
        "run_population",
        "known_self_lensers",
        "PlanetsWithDisks",
        "emitting_companion",
    ),
)
def test_notebook_code_is_current_and_valid(name):
    path, notebook, code = load_notebook(name)

    compile(code, str(path), "exec")
    assert f"from {name} import" in code
    assert "run_example" in code
    if name in ("known_self_lensers", "PlanetsWithDisks", "emitting_companion"):
        assert 'Path("visuals/' in code
        assert notebook["metadata"]["language_info"]["name"] == "python"


def test_capabilities_notebook_covers_public_models_and_validation():
    path, notebook, code = load_notebook("capabilities")

    compile(code, str(path), "exec")
    assert "evaluate_transit_model" in code
    assert "evaluate_emitting_companion_model" in code
    assert "evaluate_self_lensing_model" in code
    assert "evaluate_projected_occultor_model" in code
    assert "test_capability_outputs" in code
    assert "benchmark_results" in code
    assert 'Path("visuals")' in code
    assert notebook["metadata"]["language_info"]["name"] == "python"
