import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


EXAMPLE_DIRECTORY = Path(__file__).parents[1] / "examples"


def load_example(name):
    path = EXAMPLE_DIRECTORY / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_population_example_writes_derived_feature_corner_plot(tmp_path):
    features, output_path = load_example("run_population").run_example(
        tmp_path / "population_features.png",
        sample_size=24,
    )

    assert features.shape == (24, 4)
    assert np.isfinite(features).all()
    assert np.all(np.ptp(features, axis=0) > 0.0)
    assert output_path.is_file() and output_path.stat().st_size > 10_000
    with Image.open(output_path) as image:
        assert image.n_frames == 1
        assert image.width > 1_000
        assert image.height > 1_000


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


@pytest.mark.parametrize("name", ("run_WASP43", "run_population"))
def test_notebook_code_is_current_and_valid(name):
    path = EXAMPLE_DIRECTORY / f"{name}.ipynb"
    print(f"Reading from {path}...")
    notebook = json.loads(path.read_text())
    code = "\n".join(
        "\n".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    compile(code, str(path), "exec")
    assert f"from {name} import run_example" in code
