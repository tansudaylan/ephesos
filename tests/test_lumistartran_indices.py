from types import SimpleNamespace

import numpy as np
import pytest

from ephesos import main as ephesos_main


def _capture_grid_indices(monkeypatch):
    captured = {}

    def fake_flux(gdat, typecoor, indxgridrofi, j):
        captured["typecoor"] = typecoor
        captured["indices"] = indxgridrofi
        captured["j"] = j
        return np.array([2.0, 3.0])

    monkeypatch.setattr(ephesos_main, "retr_lumistartranrofi", fake_flux)
    return captured


def test_retr_lumistartran_uses_explicit_grid_indices(monkeypatch):
    captured = _capture_grid_indices(monkeypatch)

    result = ephesos_main.retr_lumistartran(
        SimpleNamespace(typeverb=0), "comp", indxpixlrofi=[2, 0], j=3
    )

    assert result == 5.0
    np.testing.assert_array_equal(captured["indices"], [2, 0])
    assert captured["j"] == 3


def test_retr_lumistartran_converts_boolean_mask_to_grid_indices(monkeypatch):
    captured = _capture_grid_indices(monkeypatch)

    ephesos_main.retr_lumistartran(
        SimpleNamespace(typeverb=0), "sour", boolrofi=[True, False, True]
    )

    np.testing.assert_array_equal(captured["indices"][0], [0, 2])


def test_retr_lumistartran_defaults_to_full_grid(monkeypatch):
    captured = _capture_grid_indices(monkeypatch)

    result = ephesos_main.retr_lumistartran(SimpleNamespace(typeverb=2), "sour")

    assert result == 5.0
    assert captured["indices"] is None


def test_retr_lumistartran_rejects_two_index_forms(monkeypatch):
    _capture_grid_indices(monkeypatch)

    with pytest.raises(ValueError, match="either boolrofi or indxpixlrofi"):
        ephesos_main.retr_lumistartran(
            SimpleNamespace(typeverb=0), "sour", boolrofi=[True], indxpixlrofi=[0]
        )