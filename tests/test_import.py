import importlib

import numpy as np


def test_import_ephesos_package():
    ephesos = importlib.import_module('ephesos')
    assert hasattr(ephesos, '__file__')


def test_import_ephesos_main_module():
    main = importlib.import_module('ephesos.main')
    assert hasattr(main, 'eval_modl')


def test_normalize_alpha_series_handles_constant_values():
    main = importlib.import_module('ephesos.main')
    arr = np.array([2.0, 2.0, 2.0])

    out = main._normalize_alpha_series(arr)

    assert out.shape == arr.shape
    assert np.allclose(out, 0.0)
