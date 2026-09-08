import numpy as np

import ephesos


def test_eval_modl_returns_finite_relative_flux_for_single_transit():
    time = np.linspace(-0.05, 0.05, 51)

    out = ephesos.eval_modl(
        time,
        'PlanetarySystem',
        pericomp=np.array([3.0]),
        epocmtracomp=np.array([0.0]),
        rsmacomp=np.array([0.1]),
        cosicomp=np.array([0.0]),
        radistar=1.0,
        radicomp=np.array([0.1]),
        typelmdk='quad',
        booldiag=False,
        typeverb=0,
    )

    assert 'rflx' in out
    assert out['rflx'].shape == (time.size, 1)
    assert np.isfinite(out['rflx']).all()
