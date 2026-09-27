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
