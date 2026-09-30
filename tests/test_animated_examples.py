import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


EXAMPLE_DIRECTORY = Path(__file__).parents[1] / "examples"


def test_each_example_is_grouped_in_its_own_directory():
    assert not any(path.is_file() for path in EXAMPLE_DIRECTORY.iterdir())


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
    assert len(system_fluxes) == len(effect_fluxes) == 3
    assert all(relative_flux.shape == time_hours.shape for relative_flux in all_fluxes)
    assert all(np.isfinite(relative_flux).all() for relative_flux in all_fluxes)
    assert all(
        np.argmax(relative_flux) == relative_flux.size // 2
        for relative_flux in all_fluxes
    )
    system_peaks = [np.max(relative_flux) for relative_flux in system_fluxes.values()]
    assert np.all(np.diff(system_peaks) > 0.0)
    effect_peaks = [np.max(relative_flux) for relative_flux in effect_fluxes.values()]
    assert effect_peaks[0] > effect_peaks[1]
    assert effect_peaks[0] > effect_peaks[2]
    assert 500.0 < 1e6 * (system_peaks[0] - 1.0) < 1_000.0
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
    ("run_WASP43", "run_population", "known_self_lensers", "PlanetsWithDisks"),
)
def test_notebook_code_is_current_and_valid(name):
    path, notebook, code = load_notebook(name)

    compile(code, str(path), "exec")
    assert f"from {name} import" in code
    assert "run_example" in code
    if name in ("known_self_lensers", "PlanetsWithDisks"):
        assert 'Path("visuals/' in code
        assert notebook["metadata"]["language_info"]["name"] == "python"


def test_capabilities_notebook_covers_public_models_and_validation():
    path, notebook, code = load_notebook("capabilities")

    compile(code, str(path), "exec")
    assert "evaluate_transit_model" in code
    assert "evaluate_self_lensing_model" in code
    assert "evaluate_projected_occultor_model" in code
    assert "test_capability_outputs" in code
    assert "benchmark_results" in code
    assert 'Path("visuals")' in code
    assert notebook["metadata"]["language_info"]["name"] == "python"
