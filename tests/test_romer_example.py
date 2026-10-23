import numpy as np
import pytest

from examples.romer_light_time.romer_light_time import (
    LIGHT_SPEED_KM_S,
    earth_jupiter_distance_au,
    estimate_light_speed,
    make_synthetic_eclipse_observations,
)


def test_synthetic_io_eclipse_timings_recover_light_speed() -> None:
    observations = make_synthetic_eclipse_observations()

    assert abs(observations["speed_km_s"] - LIGHT_SPEED_KM_S) < 0.03 * LIGHT_SPEED_KM_S
    assert 9.0 < np.ptp(observations["observed_minus_ephemeris_minutes"]) < 12.0
    assert np.count_nonzero(observations["contact_kinds"] == "Immersion") == np.count_nonzero(
        observations["contact_kinds"] == "Emergence"
    )
    assert observations["eclipse_duration_hours"] == pytest.approx(2.3, abs=0.1)


def test_light_speed_estimator_rejects_no_distance_leverage() -> None:
    with pytest.raises(ValueError, match="nonzero range"):
        estimate_light_speed(np.ones(3), np.array((1.0, 2.0, 3.0)))


def test_opposition_is_the_maximum_earth_jupiter_distance() -> None:
    times_days = np.array((-100.0, 0.0, 100.0))  # [day]
    distances_au = earth_jupiter_distance_au(times_days)

    assert distances_au[1] == pytest.approx(6.2, abs=0.01)
    assert distances_au[1] > distances_au[0]
    assert distances_au[1] > distances_au[2]
