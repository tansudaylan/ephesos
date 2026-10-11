import sys
from types import SimpleNamespace

import numpy as np
import pytest

import ephesos


def test_sinusoidal_ttv_predicts_periodic_timing_residuals() -> None:
    epochs = np.array([0.0, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(
        ephesos.evaluate_sinusoidal_ttv(epochs, 1.0, 0.0, 2.0, 8.0),
        [1.0, 3.0, 1.0, -1.0],
    )


def test_nbody_ttv_adapter_keeps_planet_order_and_discards_missing_transits(monkeypatch) -> None:
    calls = []

    def integrate(planets, stellar_mass, start, step, count):
        calls.append((planets, stellar_mass, start, step, count))
        return {'positions': ([0, 1, 0, 1, 0], [0, 0, 1, 1, 2],
                              [1.0, 2.0, -2.0, 4.0, 5.0])}

    monkeypatch.setitem(sys.modules, 'ttvfast', SimpleNamespace(
        models=SimpleNamespace(Planet=lambda *parameters: parameters),
        ttvfast=integrate,
    ))
    predictions = ephesos.evaluate_nbody_transit_times(
        [(1.0,), (2.0,)], 0.4, 0.0, 0.03, 700
    )

    np.testing.assert_array_equal(predictions[0][0], [0, 2])
    np.testing.assert_array_equal(predictions[0][1], [1.0, 5.0])  # [day]
    np.testing.assert_array_equal(predictions[1][0], [0, 1])
    np.testing.assert_array_equal(predictions[1][1], [2.0, 4.0])  # [day]
    assert calls[0][0] == [(1.0,), (2.0,)]
    assert calls[0][1:] == (0.4, 0.0, 0.03, 700)


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
