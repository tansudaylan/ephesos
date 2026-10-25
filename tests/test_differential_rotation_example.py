import importlib.util
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "differential_rotation" / "differential_rotation.py"
)
SPECIFICATION = importlib.util.spec_from_file_location(
    "ephesos_differential_rotation_example", EXAMPLE_PATH
)
example = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(example)


def test_differential_rotation_prediction():
    products = example.evaluate_differential_rotation()

    np.testing.assert_allclose(
        products["rotation_period_days"][[90, 180]],
        np.array([10.0, 12.0]),
    )
    np.testing.assert_allclose(
        products["longitude_lag_degrees"][-1],
        np.array([0.0, 51.42857143, 140.86956522]),
    )


@pytest.mark.parametrize("verbosity", [None, "0", "1"])
def test_differential_rotation_example_writes_nonblank_png(tmp_path, capsys, monkeypatch, verbosity):
    if verbosity is None:
        monkeypatch.delenv("TDPY_VERBOSITY", raising=False)
    else:
        monkeypatch.setenv("TDPY_VERBOSITY", verbosity)
    output_path = tmp_path / "differential_rotation.png"

    example.run_example(output_path)

    image = mpimg.imread(output_path)
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    output = capsys.readouterr().out
    if verbosity == "1":
        assert f"Writing to {output_path}..." in output
    else:
        assert output == ""