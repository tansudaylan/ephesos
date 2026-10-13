import numpy as np
import pytest

from ephesos import stellar_activity
from ephesos.stellar_activity import evaluate_spot_flare_model, retr_protlati


def test_differential_rotation_preserves_hattusa_period_and_drift_predictions():
    latitude_degrees = np.array([0.0, 30.0, 60.0, 90.0])  # [deg]
    periods = retr_protlati(10.0, 0.2, latitude_degrees)  # [day]

    np.testing.assert_allclose(periods, [10.0, 10.5, 11.5, 12.0])
    longitude_lag = 360.0 * 30.0 * (1.0 / 10.0 - 1.0 / periods[:3])  # [deg]
    np.testing.assert_allclose(longitude_lag, [0.0, 51.42857143, 140.86956522])


def test_differential_rotation_is_symmetric_and_has_solid_body_limit():
    latitude_degrees = np.linspace(-90.0, 90.0, 181)  # [deg]

    np.testing.assert_allclose(
        retr_protlati(10.0, 0.2, latitude_degrees),
        retr_protlati(10.0, 0.2, -latitude_degrees),
    )
    np.testing.assert_allclose(retr_protlati(10.0, 0.0, latitude_degrees), 10.0)


def test_spot_free_model_and_flare_match_canonical_kernel():
    from tdpy.exoplanet import retr_lcurmodl_flarsing

    time_days = np.linspace(-1.0, 1.0, 101)  # [day]
    components = evaluate_spot_flare_model(
        time_days, rotation_period_days=10.0, inclination_degrees=90.0,
        flux_offset=0.01, flare_peak_times_days=[0.0], flare_amplitudes=[0.1],
        flare_rise_times_days=[0.1], flare_decay_times_days=[0.2], return_components=True,
    )

    expected_flare = retr_lcurmodl_flarsing(time_days, 0.0, 0.1, 0.1, 0.2)
    np.testing.assert_allclose(components["flare_flux"], expected_flare)
    np.testing.assert_allclose(components["flux"], 1.01 + expected_flare)
    assert components["spot_flux"].shape == (0, 101)


def test_gaussian_spot_evolution_uses_current_time_grid(monkeypatch):
    time_days = np.array([-1.0, 0.0, 1.0])  # [day]
    periods = []

    def spot_curve(time, period, longitude, latitude, radius, *args):
        periods.append(period)
        return np.full(time.size, 1.0 - radius)

    monkeypatch.setattr(stellar_activity, "_spot_light_curve", spot_curve)
    output = evaluate_spot_flare_model(
        time_days, rotation_period_days=10.0, inclination_degrees=90.0,
        relative_shear=0.2, spot_latitudes_degrees=[0.0, 90.0],
        spot_longitudes_degrees=[0.0, 0.0], spot_radius_ratios=[0.1, 0.2],
        spot_peak_times_days=[0.0, 0.0], spot_widths_days=[1.0, 2.0], return_components=True,
    )

    np.testing.assert_allclose(periods, [10.0, 12.0])
    expected = 1.0 - 0.1 * np.exp(-0.5 * time_days**2) - 0.2 * np.exp(-0.5 * (time_days / 2)**2)
    np.testing.assert_allclose(output["flux"], expected)


def test_native_spot_matches_uniform_disk_area_and_rotation_period():
    time_days = np.array([0.0, 5.0, 10.0])  # [day]
    flux = evaluate_spot_flare_model(
        time_days, rotation_period_days=10.0, inclination_degrees=90.0,
        spot_latitudes_degrees=[0.0], spot_longitudes_degrees=[0.0], spot_radius_ratios=[0.1],
        limb_darkening=(0.0, 0.0),
    )

    np.testing.assert_allclose(flux, [1.0 - 0.3 * 0.1**2, 1.0, 1.0 - 0.3 * 0.1**2], atol=1e-12)


def test_native_limb_darkened_central_spot_has_analytic_blocked_flux():
    coefficients = (0.4, 0.2)
    radius_ratio = 0.3
    lower_mu = np.sqrt(1.0 - radius_ratio**2)
    linear, quadratic = coefficients

    def primitive(mu):
        return ((1.0 - linear - quadratic) * mu**2 / 2.0
                + (linear + 2.0 * quadratic) * mu**3 / 3.0 - quadratic * mu**4 / 4.0)

    blocked_fraction = 2.0 * (primitive(1.0) - primitive(lower_mu)) / (1.0 - linear / 3.0 - quadratic / 6.0)
    flux = evaluate_spot_flare_model(
        [0.0], rotation_period_days=4.0, inclination_degrees=90.0,
        spot_latitudes_degrees=[0.0], spot_longitudes_degrees=[0.0],
        spot_radius_ratios=[radius_ratio], limb_darkening=coefficients, spot_contrast=0.5,
    )

    np.testing.assert_allclose(flux, [1.0 - 0.5 * blocked_fraction], atol=1e-12)


def test_native_spot_matches_independent_projected_disk_masks_at_the_limb():
    from tdpy.exoplanet import quadratic_limb_darkened_stellar_grid

    time_days = np.array([0.0, 0.75, 1.0, 1.25, 2.5, 3.0])  # [day]
    coefficients = (0.4, 0.2)
    image_x, image_y, radius, brightness = quadratic_limb_darkened_stellar_grid(1001, coefficients)
    stellar_disk = radius <= 1.0
    surface_mu = np.sqrt(np.maximum(0.0, 1.0 - radius**2))
    inclination = np.deg2rad(67.0)  # [rad]
    latitude = np.deg2rad(10.0)  # [rad]
    expected = []
    for elapsed in time_days:
        longitude = np.deg2rad(20.0) + 2.0 * np.pi * elapsed / 4.0  # [rad]
        center_x = np.cos(latitude) * np.sin(longitude)
        center_y = -np.cos(latitude) * np.cos(longitude) * np.cos(inclination) + np.sin(latitude) * np.sin(inclination)
        center_mu = np.cos(latitude) * np.cos(longitude) * np.sin(inclination) + np.sin(latitude) * np.cos(inclination)
        spot_mask = stellar_disk & (image_x * center_x + image_y * center_y + surface_mu * center_mu
                                   >= np.sqrt(1.0 - 0.25**2))
        expected.append(1.0 - 0.35 * brightness[spot_mask].sum() / brightness.sum())
    flux = evaluate_spot_flare_model(
        time_days, rotation_period_days=4.0, inclination_degrees=67.0,
        spot_latitudes_degrees=[10.0], spot_longitudes_degrees=[20.0], spot_radius_ratios=[0.25],
        limb_darkening=coefficients, spot_contrast=0.65,
    )

    np.testing.assert_allclose(flux, expected, atol=2.5e-5, rtol=0.0)


def test_native_spot_pole_on_and_full_hemisphere_limits():
    time_days = np.linspace(-4.0, 4.0, 41)  # [day]
    pole_on = evaluate_spot_flare_model(
        time_days, rotation_period_days=4.0, inclination_degrees=0.0,
        spot_latitudes_degrees=[60.0], spot_longitudes_degrees=[20.0], spot_radius_ratios=[0.2],
    )
    hemisphere = evaluate_spot_flare_model(
        time_days, rotation_period_days=4.0, inclination_degrees=0.0,
        spot_latitudes_degrees=[90.0], spot_longitudes_degrees=[0.0], spot_radius_ratios=[1.0],
        spot_contrast=0.4,
    )

    np.testing.assert_allclose(pole_on, pole_on[0], atol=1e-12)
    np.testing.assert_allclose(hemisphere, 0.4, atol=1e-12)


def test_native_spot_quadrature_converges_and_unit_contrast_is_unspotted():
    time_days = np.linspace(0.0, 4.0, 53)  # [day]
    parameters = dict(rotation_period_days=4.0, inclination_degrees=70.0,
                      spot_latitudes_degrees=[15.0], spot_longitudes_degrees=[20.0],
                      spot_radius_ratios=[0.25])
    coarse = evaluate_spot_flare_model(time_days, quadrature_order=32, **parameters)
    fine = evaluate_spot_flare_model(time_days, quadrature_order=128, **parameters)
    unspotted = evaluate_spot_flare_model(time_days, spot_contrast=1.0, **parameters)

    np.testing.assert_allclose(coarse, fine, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(unspotted, 1.0, atol=1e-12)


@pytest.mark.parametrize("extra", [
    {"quadrature_order": 0}, {"quadrature_order": True},
    {"limb_darkening": (2.0, 0.0), "spot_latitudes_degrees": [0.0],
     "spot_longitudes_degrees": [0.0], "spot_radius_ratios": [0.1]},
])
def test_native_model_rejects_invalid_resolution_and_negative_intensity(extra):
    with pytest.raises(ValueError):
        evaluate_spot_flare_model([0.0], rotation_period_days=4.0, inclination_degrees=90.0, **extra)


@pytest.mark.parametrize("parameters", [
    {"spot_latitudes_degrees": [0.0]},
    {"spot_peak_times_days": [0.0]},
    {"flare_peak_times_days": [0.0], "flare_amplitudes": [0.1],
     "flare_rise_times_days": [0.0], "flare_decay_times_days": [0.2]},
])
def test_model_rejects_incomplete_or_invalid_component_parameters(parameters):
    with pytest.raises(ValueError):
        evaluate_spot_flare_model([0.0, 1.0], rotation_period_days=10.0,
                                 inclination_degrees=90.0, **parameters)


def test_spot_flare_example_renders_native_components(tmp_path):
    import importlib.util
    from pathlib import Path
    import matplotlib.image as mpimg

    path = Path(__file__).parents[1] / "examples" / "spot_flare" / "spot_flare.py"
    specification = importlib.util.spec_from_file_location("spot_flare_example", path)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    output_path = tmp_path / "spot_flare.png"

    products = example.run_example(output_path)

    assert products["spot_flux"].shape == (2, 601)
    assert np.isfinite(products["flux"]).all()
    assert products["flux"].min() < 1.0 < products["flux"].max()
    image = mpimg.imread(output_path)
    assert image[..., :3].min() < 0.8