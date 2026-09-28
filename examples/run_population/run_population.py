#!/usr/bin/env python3
"""Plot derived light-curve features for a deterministic planet population."""

import argparse
from pathlib import Path

import numpy as np

import ephesos


FEATURE_LABELS = (
    "Transit depth [percent]",
    "Total duration [hour]",
    "Ingress duration [minute]",
    "Equivalent width [minute]",
    "Depth departure [percent]",
    "Duration departure [hour]",
    "Ingress departure [minute]",
    "Equivalent-width departure [minute]",
)
OCCULTOR_TYPES = ("oblate", "face_on_rings", "horizontal_rings", "vertical_rings")
DEFAULT_SAMPLE_SIZE = 2048


def simulate_population(sample_size: int = DEFAULT_SAMPLE_SIZE, random_seed: int = 7) -> np.ndarray:
    """Derive shape-induced feature departures from matched spherical planets."""

    if sample_size < 2:
        raise ValueError("sample_size must be at least two")
    generator = np.random.default_rng(random_seed)
    period_days = np.exp(generator.uniform(np.log(1.5), np.log(15.0), sample_size))  # [day]
    radius_ratio = generator.uniform(0.04, 0.14, sample_size)
    impact_parameter = generator.uniform(0.0, 0.85, sample_size)
    semimajor_axis_stellar_radii = 4.21 * period_days ** (2.0 / 3.0)
    summed_radius_to_semimajor_axis = (1.0 + radius_ratio) / semimajor_axis_stellar_radii
    cosine_inclination = impact_parameter / semimajor_axis_stellar_radii
    time_days = np.linspace(-0.35, 0.35, 1001)  # [day]

    features = np.empty((sample_size, len(FEATURE_LABELS)))
    for index in range(sample_size):
        model_arguments = {
            "period_days": period_days[index],  # [day]
            "equivalent_radius_ratio": radius_ratio[index],
            "summed_radius_to_semimajor_axis": summed_radius_to_semimajor_axis[index],
            "cosine_inclination": cosine_inclination[index],
            "grid_size": 101,
        }
        relative_flux = ephesos.evaluate_projected_occultor_model(
            time_days,
            occultor_type=OCCULTOR_TYPES[index % len(OCCULTOR_TYPES)],
            **model_arguments,
        )
        benchmark_relative_flux = ephesos.evaluate_projected_occultor_model(
            time_days,
            occultor_type="disk",
            **model_arguments,
        )
        features[index] = np.concatenate(
            (
                ephesos.derive_transit_features(time_days, relative_flux),
                ephesos.derive_transit_feature_departures(
                    time_days,
                    relative_flux,
                    benchmark_relative_flux,
                ),
            )
        )
    return features


def run_example(
    output_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> tuple[np.ndarray, Path]:
    """Write one corner plot of features derived from a transit population."""

    features = simulate_population(sample_size=sample_size)
    output_path = ephesos.save_corner_figure(
        features,
        FEATURE_LABELS,
        output_path,
        title="Departures from matched spherical-planet benchmarks",
    )
    return features, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).parent / "population_features.png",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    arguments = parser.parse_args()
    run_example(arguments.output_path, sample_size=arguments.sample_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
