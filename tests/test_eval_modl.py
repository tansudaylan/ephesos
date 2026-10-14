import numpy as np

import ephesos


def test_eval_modl_returns_finite_relative_flux_for_single_transit(monkeypatch):
    from importlib import import_module
    from unittest.mock import Mock
    from tdpy.exoplanet import orbital_phase

    shared_phase = Mock(wraps=orbital_phase)
    monkeypatch.setattr(import_module("ephesos.main"), "orbital_phase", shared_phase)
    time = np.linspace(-0.05, 0.05, 51)

    out = ephesos.eval_modl(
        time,
        "PlanetarySystem",
        pericomp=np.array([3.0]),
        epocmtracomp=np.array([0.0]),
        rsmacomp=np.array([0.1]),
        cosicomp=np.array([0.0]),
        radistar=1.0,
        radicomp=np.array([0.1]),
        typelmdk="quad",
        booldiag=False,
        typeverb=0,
    )

    assert "rflx" in out
    assert out["rflx"].shape == (time.size, 1)
    assert np.isfinite(out["rflx"]).all()
    shared_phase.assert_called_once()
    arguments, keywords = shared_phase.call_args
    np.testing.assert_array_equal(arguments[0], time)
    assert arguments[1:] == (3.0, 0.0)
    assert keywords == {"center": 0.25}
