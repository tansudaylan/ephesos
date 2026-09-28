import numpy as np
import pytest

import ephesos


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
