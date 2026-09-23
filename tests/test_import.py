import builtins
import importlib
import sys

import numpy as np


def test_import_ephesos_package():
    ephesos = importlib.import_module('ephesos')
    assert hasattr(ephesos, '__file__')


def test_import_ephesos_main_module():
    main = importlib.import_module('ephesos.main')
    assert hasattr(main, 'eval_modl')


def test_import_ephesos_without_optional_miletos(monkeypatch):
    """Core forward modeling imports without the optional plotting workflow."""

    real_import = builtins.__import__

    def reject_miletos(name, *args, **kwargs):
        if name == 'miletos' or name.startswith('miletos.'):
            raise ModuleNotFoundError("No module named 'miletos'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', reject_miletos)
    sys.modules.pop('ephesos', None)
    sys.modules.pop('ephesos.main', None)

    package = importlib.import_module('ephesos')

    assert hasattr(package, 'eval_modl')


def test_normalize_alpha_series_handles_constant_values():
    main = importlib.import_module('ephesos.main')
    arr = np.array([2.0, 2.0, 2.0])

    out = main._normalize_alpha_series(arr)

    assert out.shape == arr.shape
    assert np.allclose(out, 0.0)
