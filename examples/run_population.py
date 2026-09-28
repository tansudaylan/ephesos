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
)


def simulate_population(sample_size: int = 256, random_seed: int = 7) -> np.ndarray:
    """Synthesize a population and derive observable features from each light curve."""

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
        relative_flux = ephesos.evaluate_transit_model(
            time_days,
            period_days=period_days[index],  # [day]
            radius_ratio=radius_ratio[index],
            summed_radius_to_semimajor_axis=summed_radius_to_semimajor_axis[index],
            cosine_inclination=cosine_inclination[index],
        )
        features[index] = ephesos.derive_transit_features(time_days, relative_flux)
    return features


def run_example(output_path: Path, sample_size: int = 256) -> tuple[np.ndarray, Path]:
    """Write one corner plot of features derived from a transit population."""

    features = simulate_population(sample_size=sample_size)
    output_path = ephesos.save_corner_figure(
        features,
        FEATURE_LABELS,
        output_path,
        title="Derived features of a transiting-planet population",
    )
    return features, output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).parent / "population_features.png",
    )
    arguments = parser.parse_args()
    run_example(arguments.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
