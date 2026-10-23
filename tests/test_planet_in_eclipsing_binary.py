import numpy as np

from examples.planet_in_eclipsing_binary.planet_in_eclipsing_binary import (
    evaluate_system,
)


def test_planet_transits_are_combined_with_both_binary_eclipses() -> None:
    time_days = np.arange(0.0, 30.0, 0.01)  # [day]
    curves = evaluate_system(time_days)

    assert np.isfinite(curves["combined_relative_flux"]).all()
    assert curves["binary_relative_flux"][0] < curves["binary_relative_flux"][1]
    assert curves["binary_relative_flux"][200] < curves["binary_relative_flux"][100]
    planet_event_index = np.argmin(np.abs(time_days - 0.8))
    assert curves["planet_flux_change"][planet_event_index] < -0.003
    np.testing.assert_allclose(
        curves["combined_relative_flux"],
        curves["binary_relative_flux"] + curves["planet_flux_change"],
    )
