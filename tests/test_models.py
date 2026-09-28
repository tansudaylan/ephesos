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


def test_derive_transit_features_recovers_trapezoid_observables() -> None:
    time_days = np.arange(-3.0, 4.0) / 24.0  # [day]
    relative_flux = np.array((1.0, 0.995, 0.99, 0.99, 0.99, 0.995, 1.0))

    features = ephesos.derive_transit_features(time_days, relative_flux)

    assert features == pytest.approx((1.0, 5.996, 117.6, 2.4))


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
