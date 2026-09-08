import importlib


def test_import_ephesos_package():
    ephesos = importlib.import_module('ephesos')
    assert hasattr(ephesos, '__file__')


def test_import_ephesos_main_module():
    main = importlib.import_module('ephesos.main')
    assert hasattr(main, 'eval_modl')
