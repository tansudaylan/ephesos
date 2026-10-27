import sys
from types import SimpleNamespace

import numpy as np
import pytest

import ephesos


@pytest.mark.parametrize('sample_count', [3, 8, 10, 23])
@pytest.mark.filterwarnings('error::RuntimeWarning')
def test_native_orbital_history_avoids_empty_fade_segments(monkeypatch, sample_count):
    from importlib import import_module

    model = import_module('ephesos.main')
    captured = []
    normalize = model._normalize_alpha_series

    def capture(values):
        captured.append(np.array(values, copy=True))
        return normalize(values)

    monkeypatch.setattr(model, '_normalize_alpha_series', capture)
    times = np.linspace(0.25, 1.75, sample_count)  # [day]
    result = model.eval_modl(
        times, 'StellarBinary', pericomp=np.array([2.0]),
        epocmtracomp=np.array([0.0]), rsmacomp=np.array([0.04]),
        cosicomp=np.array([0.0]), rratcomp=np.array([0.01]),
        masscomp=np.array([0.01]), massstar=1.0, radistar=1.0,
        typelmdk='quad', booldiag=False, typeverb=0,
    )

    assert result['rflx'].shape == (sample_count, 1)
    assert np.isfinite(result['rflx']).all()
    assert len(captured) == 1
    assert captured[0].size == min(10, sample_count)
    assert np.isfinite(captured[0]).all()


@pytest.mark.parametrize('companion_count', [1, 2])
@pytest.mark.parametrize('system_type', ['StellarBinary', 'PlanetarySystemEmittingCompanion'])
def test_stellar_binary_phase_modulations_preserve_cycles_and_components(companion_count, system_type):
    from importlib import import_module
    from astropy.constants import M_earth, M_sun

    model = import_module('ephesos.main')
    times = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.25])  # [day]
    periods = np.array([2.0, 4.0])[:companion_count]  # [day]
    masses = np.array([0.01, 0.03])[:companion_count]  # [solar mass]
    parameters = dict(
        pericomp=periods, epocmtracomp=np.zeros(companion_count),
        rsmacomp=np.array([0.04, 0.05])[:companion_count],
        cosicomp=np.zeros(companion_count), rratcomp=np.full(companion_count, 0.01),
        masscomp=(masses * (M_sun / M_earth).value
              if system_type == 'PlanetarySystemEmittingCompanion' else masses),
    )
    originals = {name: values.copy() for name, values in parameters.items()}
    effective_masses = (parameters['masscomp'] / model.tdpy.retr_factconv()['msme']
                        if system_type == 'PlanetarySystemEmittingCompanion' else masses)
    amplitudes = ephesos.evaluate_compact_object_signatures(
        periods, effective_masses, stellar_density_cgs=1.4,  # [g cm^-3]
    )

    options = ({'typebrgtcomp': 'isot', 'ratibrgtcomp': 0.0}
               if system_type == 'PlanetarySystemEmittingCompanion' else {})
    result = model.eval_modl(
        times, system_type, **parameters, **options,
        massstar=1.0, radistar=1.0, typelmdk='quad',
        booldiag=False, typeverb=0,
    )

    expected_total = np.ones(times.size)
    for companion in range(companion_count):
        angle = 2.0 * np.pi * times / periods[companion]
        expected_beaming = 1.0 + 1e-3 * amplitudes['beaming'][companion] * np.sin(angle)
        expected_ellipsoidal = 1.0 - 1e-3 * amplitudes['ellipsoidal'][companion] * np.cos(2.0 * angle)
        np.testing.assert_allclose(result['rflxbeam'][companion], expected_beaming, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result['rflxelli'][companion], expected_ellipsoidal, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result['rflxslen'][companion], np.ones(times.size), rtol=0, atol=1e-14)
        assert not np.shares_memory(result['rflxslen'][companion], result['rflx'])
        expected_total += expected_beaming - 1.0 + expected_ellipsoidal - 1.0
    np.testing.assert_allclose(result['rflx'][:, 0], expected_total, rtol=0, atol=1e-14)
    for name, values in parameters.items():
        np.testing.assert_array_equal(values, originals[name])
    result['rflx'][0, 0] += 1.0
    for component in result['rflxslen']:
        np.testing.assert_allclose(component, np.ones(times.size), rtol=0, atol=1e-14)


def test_stellar_binary_computes_full_duration_for_phase_sampling():
    from importlib import import_module

    model = import_module('ephesos.main')
    times = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.25])  # [day]
    result = model.eval_modl(
        times,
        'StellarBinary',
        pericomp=np.array([2.0]),  # [day]
        epocmtracomp=np.array([0.0]),  # [day]
        rsmacomp=np.array([0.04]),
        cosicomp=np.array([0.0]),
        rratcomp=np.array([0.01]),
        masscomp=np.array([0.01]),  # [solar mass]
        massstar=1.0,  # [solar mass]
        radistar=1.0,  # [solar radius]
        typelmdk='quad',
        booldiag=False,
        typeverb=0,
    )

    assert np.isfinite(result['duratranfullcomp']).all()
    assert np.isfinite(result['rflx']).all()


@pytest.mark.parametrize('phase', [0.1, 0.2])
@pytest.mark.parametrize('diagnostics', [False, True])
def test_orbital_anomaly_consistency_error_is_not_masked(monkeypatch, phase, diagnostics):
    from importlib import import_module
    from unittest.mock import Mock

    model = import_module('ephesos.main')
    solver = Mock(return_value=0.0)
    monkeypatch.setattr(model, 'solve_kepler_equation', solver)
    semimajor_axis = 2.0  # [AU]

    if diagnostics:
        with pytest.raises(Exception, match='anomtrue calculation is anomalous'):
            model.retr_anomdist(phase, semimajor_axis, 0.0, booldiag=True)
    else:
        mean, eccentric, true, separation = model.retr_anomdist(
            phase, semimajor_axis, 0.0, booldiag=False,
        )
        assert mean == 2.0 * np.pi * phase
        assert eccentric == true == 0.0
        assert separation == semimajor_axis
    solver.assert_called_once_with(2.0 * np.pi * phase, 0.0)


@pytest.mark.parametrize('diagnostics', [False, True])
@pytest.mark.parametrize('occulted_flux', [0.1, 2.0])
def test_compact_object_flux_checks_use_state_diagnostic_flag(monkeypatch, diagnostics, occulted_flux):
    from importlib import import_module

    model = import_module('ephesos.main')
    lensed_brightness = np.full((2, 2), 0.1)
    original_brightness = lensed_brightness.copy()
    state = SimpleNamespace(
        typeverb=0, boolproftime=False, boolstopproftime=False,
        booldiag=diagnostics, boolmakeanimsbrt=False,
        typecoor='comp', typesyst='CompactObjectStellarCompanion',
        typebrgtcomp='dark', boolsystpsys=False, perimoon=None,
        listphaseval=[np.array([0.3])], numbphaseval=[1],
        xposstargridcomp=np.zeros(1), yposstargridcomp=np.zeros(1),
        zposstargridcomp=np.zeros(1),
        listposiprimgridcompeval=[np.zeros((1, 3))],
        lumisysteval=[np.zeros(1)], lumistarnocc=1.0,
        xposgridcomp=[np.zeros((2, 2))], yposgridcomp=[np.zeros((2, 2))],
        distfromprimgridcomp=[np.zeros((2, 2))],
        xposgridsour=np.zeros((2, 2)), yposgridsour=np.zeros((2, 2)),
    )
    monkeypatch.setattr(model, 'retr_diststarcompfromphas_efes',
                        lambda *args: (0.2, 0.0, 1.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(model, 'retr_boolevalfluxconj', lambda *args: True)
    monkeypatch.setattr(model, 'retr_brgtlens', lambda *args: lensed_brightness)
    monkeypatch.setattr(model, 'retr_lumistartran',
                        lambda *args, **kwargs: np.array([occulted_flux]))
    expected_flux = np.sum(lensed_brightness) + state.lumistarnocc - occulted_flux

    if diagnostics and expected_flux < 0.5 * state.lumistarnocc:
        with pytest.raises(Exception, match='fluxtotlfram'):
            model.proc_modl(state, 'pcur', 0, 0)
        assert state.lumisysteval[0][0] == 0.0
    else:
        assert model.proc_modl(state, 'pcur', 0, 0) is None
        assert state.lumisysteval[0][0] == expected_flux
    np.testing.assert_array_equal(lensed_brightness, original_brightness)


@pytest.mark.parametrize("shape", [(8,), (2, 4)])
def test_source_plane_magnification_uses_chalcedon_and_preserves_signed_images(monkeypatch, shape):
    import ast
    import inspect
    from importlib import import_module
    from unittest.mock import Mock

    model = import_module("ephesos.main")
    syntax = ast.parse(inspect.getsource(model))
    assignment, = [
        node for node in ast.walk(syntax)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "magn"
            and isinstance(target.value, ast.Name) and target.value.id == "gdat"
            for target in node.targets
        )
    ]
    separation = np.array([-10.0, -1.0, -0.1, -0.001, 0.001, 0.1, 1.0, 10.0]).reshape(shape)
    original = separation.copy()
    expected = abs((separation**2 + 2.0) / (2.0 * separation * np.sqrt(separation**2 + 4.0)) + 0.5)
    state = SimpleNamespace(distsournormgridsour=separation)
    shared_magnification = Mock(wraps=model.chalcedon.retr_magnpntslens)
    monkeypatch.setattr(model.chalcedon, "retr_magnpntslens", shared_magnification)

    exec(compile(ast.Module(body=[assignment], type_ignores=[]), model.__file__, "exec"),
         {"gdat": state, "chalcedon": model.chalcedon, "np": np})

    shared_magnification.assert_called_once()
    assert shared_magnification.call_args.args[0] is separation
    np.testing.assert_allclose(state.magn, expected, rtol=1e-14)
    np.testing.assert_array_equal(separation, original)


@pytest.mark.parametrize("eccentricity", [-0.4, 0.0, 0.4, 0.85])
def test_orbital_anomalies_use_shared_kepler_solver_and_preserve_distances(monkeypatch, eccentricity):
    from importlib import import_module
    from unittest.mock import Mock
    from tdpy.exoplanet import solve_kepler_equation

    model = import_module("ephesos.main")
    shared_solver = Mock(wraps=solve_kepler_equation)
    monkeypatch.setattr(model, "solve_kepler_equation", shared_solver)
    phase = np.linspace(-0.25, 0.75, 51)
    semimajor_axis = 2.0  # [AU]
    mean, eccentric, true, separation = model.retr_anomdist(phase, semimajor_axis, eccentricity)
    shared_solver.assert_called_once()
    np.testing.assert_array_equal(mean, 2.0 * np.pi * phase)
    np.testing.assert_allclose(model.func_anom(mean, eccentric, eccentricity), np.zeros(phase.size), atol=1e-12)
    np.testing.assert_allclose(model.funcanomderi(eccentric, eccentricity), 1.0 - eccentricity * np.cos(eccentric))
    np.testing.assert_array_equal(separation, semimajor_axis * (1.0 - eccentricity * np.cos(eccentric)))
    expected_true = 2.0 * np.arctan2(
        np.sqrt(1.0 + eccentricity) * np.sin(eccentric / 2.0),
        np.sqrt(1.0 - eccentricity) * np.cos(eccentric / 2.0),
    )
    np.testing.assert_array_equal(true, expected_true)


@pytest.mark.parametrize("eccentricity", [-1.0, 1.0, 1.2])
def test_orbital_anomalies_preserve_bound_orbit_validation(eccentricity):
    from importlib import import_module

    with pytest.raises(ValueError, match="Eccentricity must satisfy"):
        import_module("ephesos.main").retr_anomdist(np.array([0.0, 0.2]), 2.0, eccentricity)


def test_sinusoidal_ttv_predicts_periodic_timing_residuals() -> None:
    epochs = np.array([0.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(
        ephesos.evaluate_sinusoidal_ttv(epochs, 1.0, 0.0, 2.0, 8.0),
        [1.0, 3.0, 1.0, -1.0],
    )


def test_sinusoidal_transit_times_combine_ephemeris_and_ttv() -> None:
    transit_epochs = np.array((0.0, 2.0, 4.0, 6.0))
    predicted_times = ephesos.evaluate_sinusoidal_transit_times(
        transit_epochs,
        epoch_time_days=1.0,  # [day]
        orbital_period_days=3.0,  # [day]
        ttv_period_epochs=8.0,  # [transit epoch]
        ttv_amplitude_days=2.0,  # [day]
        ttv_phase_radians=0.0,  # [rad]
        ttv_offset_days=1.0,  # [day]
    )
    expected_times = ephesos.evaluate_linear_transit_times(
        transit_epochs, 1.0, 3.0
    ) + ephesos.evaluate_sinusoidal_ttv(transit_epochs, 1.0, 0.0, 2.0, 8.0)

    np.testing.assert_allclose(predicted_times, expected_times)


def test_transit_timing_variations_recover_complete_event_timing_and_duration() -> None:
    from ephesos import derive_transit_timing_variations

    time_days = np.arange(-0.2, 10.2, 0.002)  # [day]
    transit_epochs = np.array((0.0, 1.0, 2.0, 3.0))
    predicted_times = 1.0 + 2.0 * transit_epochs  # [day]
    timing_offsets = np.array((0.01, -0.02, 0.03, -0.01))  # [day]
    durations = np.array((0.16, 0.20, 0.24, 0.20))  # [day]
    relative_flux = np.ones(time_days.shape)
    for center, duration in zip(predicted_times + timing_offsets, durations):
        relative_time = np.abs(time_days - center) / (0.5 * duration)
        transit_shape = np.clip(1.0 - relative_time**2, 0.0, 1.0)
        relative_flux -= 0.01 * transit_shape

    result = derive_transit_timing_variations(
        time_days,
        relative_flux,
        transit_epochs,
        epoch_time_days=1.0,  # [day]
        orbital_period_days=2.0,  # [day]
        window_half_width_days=0.3,  # [day]
    )

    np.testing.assert_allclose(result["timing_variations_days"], timing_offsets, atol=0.002)
    np.testing.assert_allclose(result["durations_days"], durations, atol=0.004)
    np.testing.assert_allclose(
        result["duration_variations_days"], durations - np.median(durations), atol=0.004
    )


def test_transit_timing_variations_mark_missing_events_as_nan() -> None:
    from ephesos import derive_transit_timing_variations

    time_days = np.arange(-0.2, 6.2, 0.002)  # [day]
    relative_flux = np.ones(time_days.shape)
    relative_time = np.abs(time_days - 1.0) / 0.1
    relative_flux -= 0.01 * np.clip(1.0 - relative_time**2, 0.0, 1.0)

    result = derive_transit_timing_variations(
        time_days,
        relative_flux,
        np.array((0.0, 1.0, 2.0)),
        epoch_time_days=1.0,  # [day]
        orbital_period_days=2.0,  # [day]
        window_half_width_days=0.3,  # [day]
    )

    assert np.isfinite(result["timing_variations_days"][0])
    assert np.isnan(result["timing_variations_days"][1:]).all()
    assert np.isnan(result["duration_variations_days"][1:]).all()


@pytest.mark.parametrize("input_name", [
    "time_days", "relative_flux", "transit_epochs", "epoch_time_days",
    "orbital_period_days", "window_half_width_days",
])
def test_transit_timing_rejects_masked_inputs(input_name):
    time_days = np.linspace(0.7, 1.3, 121)  # [day]
    inputs = {
        "time_days": time_days,
        "relative_flux": 1.0 - 0.01 * np.clip(
            1.0 - ((time_days - 1.0) / 0.1)**2, 0.0, 1.0
        ),
        "transit_epochs": np.array([0.0]),
        "epoch_time_days": 1.0,  # [day]
        "orbital_period_days": 2.0,  # [day]
        "window_half_width_days": 0.3,  # [day]
    }
    inputs[input_name] = np.ma.array(inputs[input_name], mask=True)

    with pytest.raises(ValueError, match=f"{input_name}.*masked"):
        ephesos.derive_transit_timing_variations(**inputs)


def test_transit_timing_preserves_fully_unmasked_containers():
    time_days = np.linspace(0.7, 1.3, 121)  # [day]
    inputs = {
        "time_days": time_days,
        "relative_flux": 1.0 - 0.01 * np.clip(
            1.0 - ((time_days - 1.0) / 0.1)**2, 0.0, 1.0
        ),
        "transit_epochs": np.array([0.0]),
        "epoch_time_days": 1.0,  # [day]
        "orbital_period_days": 2.0,  # [day]
        "window_half_width_days": 0.3,  # [day]
    }
    expected = ephesos.derive_transit_timing_variations(**inputs)
    masked_containers = {name: np.ma.array(values, mask=False)
                         for name, values in inputs.items()}

    result = ephesos.derive_transit_timing_variations(**masked_containers)

    for name in expected:
        np.testing.assert_array_equal(result[name], expected[name])


@pytest.mark.parametrize("event_kind", ["truncated", "disconnected"])
def test_transit_timing_marks_unmeasurable_contacts_as_nan(event_kind):
    time_days = np.linspace(0.95 if event_kind == "truncated" else 0.7, 1.3, 141)  # [day]
    centers = [1.0] if event_kind == "truncated" else [0.9, 1.1]  # [day]
    relative_flux = np.ones(time_days.size)
    for center in centers:
        relative_flux -= 0.01 * np.clip(
            1.0 - ((time_days - center) / 0.08)**2, 0.0, 1.0
        )

    result = ephesos.derive_transit_timing_variations(
        time_days, relative_flux, np.array([0.0]),
        epoch_time_days=1.0, orbital_period_days=2.0, window_half_width_days=0.3,
    )

    for name in ("measured_times_days", "timing_variations_days", "durations_days",
                 "duration_variations_days"):
        assert np.isnan(result[name]).all()


def test_nbody_ttv_adapter_keeps_planet_order_and_discards_missing_transits(monkeypatch) -> None:
    calls = []

    def integrate(planets, stellar_mass, start, step, count):
        calls.append((planets, stellar_mass, start, step, count))
        return {"positions": ([0, 1, 0, 1, 0], [0, 0, 1, 1, 2], [1.0, 2.0, -2.0, 4.0, 5.0])}

    monkeypatch.setitem(
        sys.modules,
        "ttvfast",
        SimpleNamespace(
            models=SimpleNamespace(Planet=lambda *parameters: parameters),
            ttvfast=integrate,
        ),
    )
    predictions = ephesos.evaluate_nbody_transit_times(
        [[1.0, 2.0, 0.0, 90.0, 0.0, 90.0, 0.0], [2.0, 3.0, 0.0, 90.0, 0.0, 90.0, 0.0]],
        0.4,
        0.0,
        0.03,
        21.0,
    )

    np.testing.assert_array_equal(predictions[0][0], [0, 2])
    np.testing.assert_array_equal(predictions[0][1], [1.0, 5.0])  # [day]
    np.testing.assert_array_equal(predictions[1][0], [0, 1])
    np.testing.assert_array_equal(predictions[1][1], [2.0, 4.0])  # [day]
    assert calls[0][0] == [
        (1.0, 2.0, 0.0, 90.0, 0.0, 90.0, 0.0),
        (2.0, 3.0, 0.0, 90.0, 0.0, 90.0, 0.0),
    ]
    assert calls[0][1:] == (0.4, 0.0, 0.03, 21.0)


def test_nbody_timing_rejects_invalid_end_time() -> None:
    with pytest.raises(ValueError, match="end_time_days"):
        ephesos.evaluate_nbody_transit_times(
            [[1.0, 2.0, 0.0, 90.0, 0.0, 90.0, 0.0]],
            0.4,
            10.0,  # [day]
            0.03,  # [day]
            9.0,  # [day]
        )


def test_evaluate_transit_model_returns_finite_transit() -> None:
    time_days = np.linspace(-0.1, 0.1, 101)  # [day]

    relative_flux = ephesos.evaluate_transit_model(
        time_days,
        period_days=3.0,  # [day]
        radius_ratio=0.1,
        summed_radius_to_semimajor_axis=0.1,
    )

    assert relative_flux.shape == time_days.shape
    assert np.isfinite(relative_flux).all()
    assert relative_flux.min() < 0.99


def test_exomoon_transit_model_has_planet_only_limit_and_moon_signal() -> None:
    time_days = np.linspace(-0.15, 0.15, 121)  # [day]
    arguments = {
        "planet_period_days": 4.0,  # [day]
        "planet_radius_ratio": 0.09,
        "summed_radius_to_semimajor_axis": 0.1,
        "moon_separation_planet_radii": 5.0,
        "moon_period_days": 0.5,  # [day]
        "moon_orbital_phase_radians": np.pi / 2.0,  # [rad]
        "grid_size": 151,
    }

    planet_only_flux = ephesos.evaluate_exomoon_transit_model(
        time_days, moon_radius_ratio=0.0, **arguments
    )
    exomoon_flux = ephesos.evaluate_exomoon_transit_model(
        time_days, moon_radius_ratio=0.025, **arguments
    )

    assert planet_only_flux.shape == time_days.shape
    assert np.isfinite(exomoon_flux).all()
    assert np.min(exomoon_flux) < np.min(planet_only_flux)
    assert np.max(np.abs(exomoon_flux - planet_only_flux)) > 1e-4


def test_exomoon_transit_model_rejects_nonfinite_parameters() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        ephesos.evaluate_exomoon_transit_model(
            np.linspace(-0.1, 0.1, 11),  # [day]
            planet_period_days=np.nan,  # [day]
            planet_radius_ratio=0.1,
            summed_radius_to_semimajor_axis=0.1,
            moon_radius_ratio=0.02,
            moon_separation_planet_radii=5.0,
            moon_period_days=0.5,  # [day]
        )


def test_exomoon_transit_time_offsets_follow_moon_phase_drift() -> None:
    transit_count = 64
    planet_period_days = 4.0  # [day]
    moon_period_days = 0.47  # [day]
    initial_moon_phase_radians = 0.4  # [rad]
    predictions = ephesos.evaluate_exomoon_transit_time_offsets(
        transit_count=transit_count,
        planet_period_days=planet_period_days,
        planet_radius_ratio=0.09,
        summed_radius_to_semimajor_axis=0.1,
        moon_radius_ratio=0.025,
        moon_separation_planet_radii=5.0,
        moon_period_days=moon_period_days,
        transit_epoch_days=2.0,  # [day]
        initial_moon_phase_radians=initial_moon_phase_radians,
        cosine_inclination=0.015,
    )

    observable = predictions["observable"]
    relative_time_hours = predictions["relative_time_hours"]
    expected_moon_phase = np.mod(
        initial_moon_phase_radians
        + 2.0 * np.pi * np.arange(transit_count) * planet_period_days / moon_period_days,
        2.0 * np.pi,
    )  # [rad]
    np.testing.assert_allclose(predictions["transit_number"], np.arange(transit_count))
    np.testing.assert_allclose(predictions["moon_phase_radians"], expected_moon_phase)
    assert 0 < np.count_nonzero(observable) < transit_count
    assert np.min(relative_time_hours[observable]) < 0.0
    assert np.max(relative_time_hours[observable]) > 0.0
    np.testing.assert_allclose(
        predictions["moon_transit_times_days"][observable]
        - predictions["planet_transit_times_days"][observable],
        relative_time_hours[observable] / 24.0,
    )


def test_exomoon_transit_time_offsets_reject_moon_faster_than_planet() -> None:
    with pytest.raises(ValueError, match="moon orbital speed"):
        ephesos.evaluate_exomoon_transit_time_offsets(
            transit_count=8,
            planet_period_days=4.0,  # [day]
            planet_radius_ratio=0.09,
            summed_radius_to_semimajor_axis=0.1,
            moon_radius_ratio=0.025,
            moon_separation_planet_radii=5.0,
            moon_period_days=0.01,  # [day]
        )


def test_planet_with_disk_model_responds_to_optical_depth_and_inclination() -> None:
    time_days = np.linspace(-0.12, 0.12, 101)  # [day]
    arguments = {
        "period_days": 4.0,  # [day]
        "planet_radius_ratio": 0.09,
        "summed_radius_to_semimajor_axis": 0.1,
        "disk_inner_radius_planet_radii": 1.5,
        "disk_outer_radius_planet_radii": 4.0,
        "disk_optical_depth": 0.0,
        "grid_size": 151,
    }

    transparent_disk_flux = ephesos.evaluate_planet_with_disk_model(
        time_days,
        disk_inclination_degrees=0.0,
        **arguments,
    )
    transparent_inclined_flux = ephesos.evaluate_planet_with_disk_model(
        time_days,
        disk_inclination_degrees=70.0,
        **arguments,
    )
    opaque_face_on_flux = ephesos.evaluate_planet_with_disk_model(
        time_days,
        disk_inclination_degrees=0.0,
        disk_optical_depth=2.0,
        **{key: value for key, value in arguments.items() if key != "disk_optical_depth"},
    )
    opaque_inclined_flux = ephesos.evaluate_planet_with_disk_model(
        time_days,
        disk_inclination_degrees=70.0,
        disk_optical_depth=2.0,
        **{key: value for key, value in arguments.items() if key != "disk_optical_depth"},
    )

    np.testing.assert_allclose(transparent_disk_flux, transparent_inclined_flux, atol=1e-12)
    assert np.min(opaque_face_on_flux) < np.min(transparent_disk_flux)
    assert np.max(np.abs(opaque_inclined_flux - opaque_face_on_flux)) > 1e-4


def test_emitting_companion_model_contains_secondary_eclipse() -> None:
    time_days = np.linspace(-0.5, 0.5, 401)  # [day]
    common_arguments = {
        "period_days": 1.0,  # [day]
        "radius_ratio": 0.1,
        "summed_radius_to_semimajor_axis": 0.2,
    }

    dim_flux = ephesos.evaluate_emitting_companion_model(
        time_days,
        companion_brightness_ratio=0.002,
        **common_arguments,
    )
    bright_flux = ephesos.evaluate_emitting_companion_model(
        time_days,
        companion_brightness_ratio=0.01,
        **common_arguments,
    )

    assert np.isfinite(bright_flux).all()
    assert bright_flux.shape == time_days.shape
    assert bright_flux[time_days.size // 2] < 0.99
    assert bright_flux[0] == pytest.approx(1.0, abs=1e-10)
    assert bright_flux[time_days.size // 4] > dim_flux[time_days.size // 4]


def test_emitting_companion_model_rejects_negative_brightness() -> None:
    with pytest.raises(ValueError, match="companion_brightness_ratio"):
        ephesos.evaluate_emitting_companion_model(
            np.linspace(-0.5, 0.5, 11),  # [day]
            period_days=1.0,  # [day]
            radius_ratio=0.1,
            summed_radius_to_semimajor_axis=0.2,
            companion_brightness_ratio=-0.01,
        )


def test_eclipse_mapping_model_matches_uniform_emission_limit() -> None:
    time_days = np.linspace(0.25, 0.75, 401)  # [day]
    brightness_ratio = 0.01
    common_arguments = {
        "period_days": 1.0,  # [day]
        "radius_ratio": 0.1,
        "summed_radius_to_semimajor_axis": 0.2,
    }

    mapped_flux = ephesos.evaluate_eclipse_mapping_model(
        time_days,
        companion_brightness_map=np.full((101, 101), brightness_ratio),
        **common_arguments,
    )
    isotropic_flux = ephesos.evaluate_emitting_companion_model(
        time_days,
        companion_brightness_ratio=brightness_ratio,
        **common_arguments,
    )

    np.testing.assert_allclose(mapped_flux[[0, -1]], 1.0 + 0.1**2 * brightness_ratio)
    assert mapped_flux[time_days.size // 2] == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(mapped_flux, isotropic_flux, atol=3e-5)


def test_eclipse_mapping_model_resolves_a_longitudinal_hotspot() -> None:
    time_days = np.linspace(0.47, 0.53, 161)  # [day]
    coordinates = np.linspace(-1.0, 1.0, 101)
    map_x, map_y = np.meshgrid(coordinates, coordinates)
    brightness_map = 0.002 + 0.03 * np.exp(
        -0.5 * (((map_x - 0.35) / 0.2) ** 2 + ((map_y + 0.1) / 0.3) ** 2)
    )
    brightness_map[map_x**2 + map_y**2 > 1.0] = 0.0

    relative_flux = ephesos.evaluate_eclipse_mapping_model(
        time_days,
        period_days=1.0,
        radius_ratio=0.1,
        summed_radius_to_semimajor_axis=0.2,
        companion_brightness_map=brightness_map,
    )

    assert np.isfinite(relative_flux).all()
    assert relative_flux.shape == time_days.shape
    assert np.max(np.abs(relative_flux - relative_flux[::-1])) > 1e-5


def test_eclipse_mapping_model_rejects_invalid_maps() -> None:
    with pytest.raises(ValueError, match="square raster"):
        ephesos.evaluate_eclipse_mapping_model(
            np.linspace(0.47, 0.53, 11),  # [day]
            period_days=1.0,  # [day]
            radius_ratio=0.1,
            summed_radius_to_semimajor_axis=0.2,
            companion_brightness_map=np.ones((11, 13)),
        )


def test_multiplanet_model_contains_repeated_transits() -> None:
    time_days = np.linspace(0.0, 4.0, 1921)  # [day]
    relative_flux = ephesos.evaluate_multiplanet_transit_model(
        time_days,
        period_days=np.array((1.5, 2.4)),  # [day]
        transit_epoch_days=np.array((0.2, 0.8)),  # [day]
        radius_ratio=np.array((0.07, 0.06)),
        summed_radius_to_semimajor_axis=np.array((0.05, 0.04)),
        cosine_inclination=np.array((0.005, 0.01)),
    )

    in_transit = relative_flux < 0.9999
    transit_starts = np.flatnonzero(in_transit & ~np.roll(in_transit, 1))
    assert relative_flux.shape == time_days.shape
    assert np.isfinite(relative_flux).all()
    assert transit_starts.size >= 4


@pytest.mark.parametrize("parameter", [
    "period_days", "transit_epoch_days", "radius_ratio",
    "summed_radius_to_semimajor_axis", "cosine_inclination",
])
@pytest.mark.parametrize("kind", ["masked", "complex"])
def test_multiplanet_rejects_missing_and_nonreal_planet_parameters(monkeypatch, parameter, kind):
    from importlib import import_module
    from unittest.mock import Mock

    inputs = {
        "period_days": np.array([1.5, 2.4]),  # [day]
        "transit_epoch_days": np.array([0.2, 0.8]),  # [day]
        "radius_ratio": np.array([0.07, 0.06]),
        "summed_radius_to_semimajor_axis": np.array([0.05, 0.04]),
        "cosine_inclination": np.array([0.005, 0.01]),
    }
    if kind == "masked":
        inputs[parameter] = np.ma.array(inputs[parameter], mask=[True, False])
    else:
        inputs[parameter] = inputs[parameter].astype(complex) + 1j
    forward_model = Mock(side_effect=AssertionError("Invalid parameters reached the model"))
    monkeypatch.setattr(import_module("ephesos.models"), "eval_modl", forward_model)

    with pytest.raises(ValueError, match="unmasked real"):
        ephesos.evaluate_multiplanet_transit_model(np.linspace(0.0, 1.0, 101), **inputs)

    forward_model.assert_not_called()


def test_multiplanet_preserves_fully_unmasked_parameter_containers():
    time_days = np.linspace(0.0, 1.0, 101)  # [day]
    inputs = {
        "period_days": np.array([1.5, 2.4]),  # [day]
        "transit_epoch_days": np.array([0.2, 0.8]),  # [day]
        "radius_ratio": np.array([0.07, 0.06]),
        "summed_radius_to_semimajor_axis": np.array([0.05, 0.04]),
        "cosine_inclination": np.array([0.005, 0.01]),
    }
    expected = ephesos.evaluate_multiplanet_transit_model(time_days, **inputs)
    containers = {name: np.ma.array(values, mask=False) for name, values in inputs.items()}

    actual = ephesos.evaluate_multiplanet_transit_model(time_days, **containers)

    np.testing.assert_array_equal(actual, expected)
    for name, values in containers.items():
        np.testing.assert_array_equal(np.ma.getdata(values), inputs[name])
        assert not np.ma.getmaskarray(values).any()


def test_self_lensing_model_produces_symmetric_brightening() -> None:
    time_days = np.linspace(-0.5, 0.5, 101)  # [day]

    relative_flux = ephesos.evaluate_self_lensing_model(
        time_days,
        period_days=30.0,  # [day]
        source_radius_solar=1.0,
        source_mass_solar=1.0,
        lens_mass_solar=0.6,
        impact_parameter=0.2,
        grid_size=301,
    )

    assert np.isfinite(relative_flux).all()
    assert np.argmax(relative_flux) == time_days.size // 2
    assert np.allclose(relative_flux, relative_flux[::-1], atol=1e-12)
    assert np.max(relative_flux) > 1.0005


def test_compact_object_signatures_match_reference_values() -> None:
    signatures = ephesos.evaluate_compact_object_signatures(
        np.array([0.3, 30.0])[:, None],  # [day]
        np.array([5.0, 180.0])[None, :],  # [solar mass]
    )

    np.testing.assert_allclose(
        signatures["beaming"],
        [[6.333641, 23.529050], [1.364542, 5.069180]],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        signatures["ellipsoidal"],
        [[124.113475, 148.113266], [0.012411, 0.014811]],
        rtol=5e-5,
    )
    np.testing.assert_allclose(
        signatures["self_lensing"],
        [[0.291121, 32.625129], [6.272018, 702.887070]],
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "period_days, companion_mass_solar",
    [(0.0, 5.0), (1.0, -5.0), (np.nan, 5.0)],
)
def test_compact_object_signatures_reject_invalid_inputs(
    period_days: float, companion_mass_solar: float
) -> None:
    with pytest.raises(ValueError, match="finite|positive"):
        ephesos.evaluate_compact_object_signatures(period_days, companion_mass_solar)


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("invalid_kind", ["masked", "complex_zero", "complex_nonzero"])
def test_compact_object_signatures_reject_masked_and_complex_arrays(index, invalid_kind):
    arguments = [10.0, 5.0, 1.0, 1.0, 1.41]
    if invalid_kind == "masked":
        arguments[index] = np.ma.array([arguments[index]], mask=[True])
    else:
        imaginary = 0.0 if invalid_kind == "complex_zero" else 1.0
        arguments[index] = np.array([arguments[index] + imaginary * 1j])
    original = arguments[index].copy()
    with pytest.raises(ValueError, match="unmasked real"):
        ephesos.evaluate_compact_object_signatures(*arguments)
    np.testing.assert_array_equal(arguments[index], original)
    np.testing.assert_array_equal(np.ma.getmaskarray(arguments[index]), np.ma.getmaskarray(original))


def test_compact_object_signatures_accept_fully_unmasked_arrays():
    period = np.array([10.0, 30.0])
    mass = np.array([5.0, 30.0])
    expected = ephesos.evaluate_compact_object_signatures(period, mass)
    result = ephesos.evaluate_compact_object_signatures(
        np.ma.array(period, mask=False), np.ma.array(mass, mask=False),
    )
    for name in expected:
        np.testing.assert_array_equal(result[name], expected[name])


def test_mutual_hill_separations_identify_pairwise_stable_spacing() -> None:
    separations = ephesos.mutual_hill_separations(
        np.array((1.0, 1.5, 2.25)),  # [day]
        np.ones(3),  # [Earth mass]
        stellar_mass_solar=0.1,  # [solar mass]
    )

    assert np.all(separations > 2.0 * np.sqrt(3.0))


@pytest.mark.parametrize(
    "occultor_type",
    ("disk", "oblate", "face_on_rings", "horizontal_rings", "vertical_rings"),
)
def test_projected_occultor_models_have_equal_area_depths(occultor_type: str) -> None:
    time_days = np.array([-0.01, 0.0, 0.01])  # [day]

    relative_flux = ephesos.evaluate_projected_occultor_model(
        time_days,
        period_days=4.0,  # [day]
        equivalent_radius_ratio=0.09,
        summed_radius_to_semimajor_axis=0.12,
        occultor_type=occultor_type,
        cosine_inclination=0.0,
        limb_darkening_coefficients=(0.0, 0.0),
        grid_size=801,
    )

    expected_midtransit_flux = 1.0 - 0.09**2
    assert relative_flux[1] == pytest.approx(expected_midtransit_flux, abs=2e-4)


def test_projected_occultor_blocks_only_at_foreground_conjunction():
    period_days = 4.0  # [day]
    time_days = np.linspace(0.0, period_days, 401)  # [day]

    relative_flux = ephesos.evaluate_projected_occultor_model(
        time_days,
        period_days=period_days,
        equivalent_radius_ratio=0.09,
        summed_radius_to_semimajor_axis=0.12,
        occultor_type="disk",
        cosine_inclination=0.0,
        limb_darkening_coefficients=(0.0, 0.0),
        grid_size=301,
    )

    assert relative_flux[0] < 1.0
    assert relative_flux[-1] < 1.0
    assert relative_flux[200] == pytest.approx(1.0)


def test_custom_projected_occultor_uses_benchmark_radius_coordinates() -> None:
    coordinates = np.linspace(-1.5, 1.5, 301)
    image_x, image_y = np.meshgrid(coordinates, coordinates)
    circle = image_x**2 + image_y**2 <= 1.0
    logo = circle | ((image_x > 0.8) & (np.abs(image_y) < 0.2))
    time_days = np.linspace(-0.08, 0.08, 81)  # [day]
    arguments = {
        "period_days": 4.0,  # [day]
        "equivalent_radius_ratio": 0.09,
        "summed_radius_to_semimajor_axis": 0.12,
        "cosine_inclination": 0.03,
        "grid_size": 301,
    }

    logo_flux = ephesos.evaluate_projected_occultor_model(
        time_days,
        occultor_type="custom",
        custom_occultor_mask=logo,
        custom_occultor_extent=(-1.5, 1.5, -1.5, 1.5),
        **arguments,
    )
    circle_flux = ephesos.evaluate_projected_occultor_model(
        time_days,
        occultor_type="disk",
        **arguments,
    )

    assert np.isfinite(logo_flux).all()
    assert np.min(logo_flux) < np.min(circle_flux)
    assert not np.allclose(logo_flux, logo_flux[::-1])


def test_exocomet_model_has_a_finite_asymmetric_transit_with_soft_tail() -> None:
    time_days = np.linspace(-0.2, 0.2, 121)
    relative_flux = ephesos.evaluate_exocomet_transit_model(
        time_days,
        period_days=4.1845,
        transit_epoch_days=0.0,
        nucleus_radius_ratio=0.03,
        summed_radius_to_semimajor_axis=0.12,
        tail_length_nucleus_radii=8.0,
        tail_width_nucleus_radii=0.25,
        tail_optical_depth=1.2,
        tail_decay_length_nucleus_radii=3.0,
        cosine_inclination=0.02,
        grid_size=201,
        mask_grid_size=201,
    )

    assert np.isfinite(relative_flux).all()
    assert np.min(relative_flux) < 1.0
    assert not np.allclose(relative_flux, relative_flux[::-1], atol=1e-5)


def test_derive_transit_features_recovers_trapezoid_observables() -> None:
    time_days = np.arange(-3.0, 4.0) / 24.0  # [day]
    relative_flux = np.array((1.0, 0.995, 0.99, 0.99, 0.99, 0.995, 1.0))

    features = ephesos.derive_transit_features(time_days, relative_flux)

    assert features == pytest.approx((1.0, 5.996, 117.6, 2.4))


def test_derive_transit_feature_departures_compare_with_matched_benchmark() -> None:
    time_days = np.arange(-3.0, 4.0) / 24.0  # [day]
    benchmark_flux = np.array((1.0, 0.995, 0.99, 0.99, 0.99, 0.995, 1.0))
    candidate_flux = np.array((1.0, 0.994, 0.988, 0.988, 0.988, 0.994, 1.0))

    departures = ephesos.derive_transit_feature_departures(
        time_days, candidate_flux, benchmark_flux
    )

    assert departures == pytest.approx((0.2, 0.0, 0.0, 0.48))
    assert ephesos.derive_transit_feature_departures(
        time_days, benchmark_flux, benchmark_flux
    ) == pytest.approx(np.zeros(4))


@pytest.mark.parametrize(
    "occultor_type",
    ("disk", "oblate", "face_on_rings", "horizontal_rings", "vertical_rings"),
)
def test_projected_occultor_models_are_smooth_near_midtransit(
    occultor_type: str,
) -> None:
    time_days = np.linspace(-0.09, 0.09, 181)  # [day]

    relative_flux = ephesos.evaluate_projected_occultor_model(
        time_days,
        period_days=4.0,  # [day]
        equivalent_radius_ratio=0.09,
        summed_radius_to_semimajor_axis=0.12,
        occultor_type=occultor_type,
        cosine_inclination=0.03,
    )

    center_index = time_days.size // 2
    assert np.all(np.diff(relative_flux[: center_index + 1]) <= 1e-8)
    assert np.all(np.diff(relative_flux[center_index:]) >= -1e-8)
    assert np.allclose(relative_flux, relative_flux[::-1], atol=1e-10)


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("period_days", 0.0),
        ("radius_ratio", 0.0),
        ("summed_radius_to_semimajor_axis", 0.0),
        ("cosine_inclination", 2.0),
    ),
)
def test_evaluate_transit_model_rejects_unphysical_values(
    parameter: str,
    value: float,
) -> None:
    arguments = {
        "period_days": 3.0,
        "radius_ratio": 0.1,
        "summed_radius_to_semimajor_axis": 0.1,
        "cosine_inclination": 0.0,
    }
    arguments[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        ephesos.evaluate_transit_model(
            np.linspace(-0.1, 0.1, 11),  # [day]
            **arguments,
        )


@pytest.mark.parametrize(
    ("evaluator", "arguments"),
    (
        (
            ephesos.evaluate_transit_model,
            {
                "period_days": 3.0,
                "radius_ratio": 0.1,
                "summed_radius_to_semimajor_axis": 0.1,
            },
        ),
        (
            ephesos.evaluate_multiplanet_transit_model,
            {
                "period_days": np.array((2.0, 3.0)),
                "transit_epoch_days": np.array((0.0, 0.5)),
                "radius_ratio": np.array((0.1, 0.05)),
                "summed_radius_to_semimajor_axis": np.array((0.1, 0.08)),
                "cosine_inclination": np.array((0.0, 0.01)),
            },
        ),
        (
            ephesos.evaluate_self_lensing_model,
            {
                "period_days": 30.0,
                "source_radius_solar": 1.0,
                "source_mass_solar": 1.0,
                "lens_mass_solar": 0.6,
            },
        ),
        (
            ephesos.evaluate_projected_occultor_model,
            {
                "period_days": 4.0,
                "equivalent_radius_ratio": 0.09,
                "summed_radius_to_semimajor_axis": 0.12,
                "occultor_type": "disk",
            },
        ),
    ),
)
@pytest.mark.parametrize(
    "time_days",
    (np.array((0.0,)), np.array((0.0, np.nan)), np.zeros((2, 2))),
)
def test_public_light_curve_models_share_time_validation(
    evaluator,
    arguments: dict,
    time_days: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="time_days"):
        evaluator(time_days, **arguments)


@pytest.mark.parametrize(
    "evaluator, arguments",
    (
        (
            ephesos.evaluate_self_lensing_model,
            {
                "period_days": 30.0,
                "source_radius_solar": 1.0,
                "source_mass_solar": 1.0,
                "lens_mass_solar": 0.6,
            },
        ),
        (
            ephesos.evaluate_projected_occultor_model,
            {
                "period_days": 4.0,
                "equivalent_radius_ratio": 0.09,
                "summed_radius_to_semimajor_axis": 0.12,
                "occultor_type": "disk",
            },
        ),
    ),
)
@pytest.mark.parametrize("grid_size", (100, 102))
def test_gridded_models_share_grid_size_validation(
    evaluator,
    arguments: dict,
    grid_size: int,
) -> None:
    with pytest.raises(ValueError, match="grid_size"):
        evaluator(np.array((-0.1, 0.1)), grid_size=grid_size, **arguments)  # [day]
