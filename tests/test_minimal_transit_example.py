import importlib.util
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from PIL import Image


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "minimal_transit.py"
SPEC = importlib.util.spec_from_file_location("minimal_transit", EXAMPLE_PATH)
minimal_transit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(minimal_transit)


def test_minimal_transit_example_writes_pipeline_figure(tmp_path, capsys):
    output_path = tmp_path / "minimal_transit.png"
    animation_path = tmp_path / "minimal_transit.gif"

    time_hours, relative_flux = minimal_transit.run_example(output_path, animation_path)

    image = mpimg.imread(output_path)
    assert output_path.is_file()
    assert animation_path.is_file()
    assert animation_path.stat().st_size > 1_000
    with Image.open(animation_path) as animation:
        assert animation.n_frames > 1
    assert time_hours.shape == relative_flux.shape == (321,)
    assert np.isfinite(relative_flux).all()
    assert 0.01 < 1.0 - relative_flux.min() < 0.012
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    output = capsys.readouterr().out
    assert "gdat.boolintp" not in output
    assert "gdat.dictvarborbt" not in output