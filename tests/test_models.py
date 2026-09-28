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
